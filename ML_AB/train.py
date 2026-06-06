import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from ML_AB.actions import action_features_torch
from ML_AB.agents import ModelAgent
from ML_AB.data import Replay, collect_episode
from ML_AB.models import Big2TransformerNet, checkpoint_payload
from ML_AB.state import HISTORY_LEN
from ML_AB.utils import configure_torch_threads, device_from_arg, set_seed


def adapt_action_feature_state(state, target_state):
    state = dict(state)
    for key in ("action_proj.0.weight", "action_bias.weight"):
        if key not in state or key not in target_state:
            continue
        src = state[key]
        dst = target_state[key]
        if src.shape == dst.shape:
            continue
        if src.ndim != 2 or dst.ndim != 2 or src.shape[0] != dst.shape[0]:
            continue
        adapted = dst.new_zeros(dst.shape)
        width = min(src.shape[1], dst.shape[1])
        adapted[:, :width] = src[:, :width]
        state[key] = adapted
    return state


def load_compatible_state(model, state):
    state = adapt_action_feature_state(state, model.state_dict())
    target_keys = set(model.state_dict())
    state_keys = set(state)
    strict = ("history_pos" in state) and target_keys.issubset(state_keys)
    model.load_state_dict(state, strict=strict)


def batch_to_tensors(batch, device):
    card = torch.tensor(np.array([b["card_feats"] for b in batch]), dtype=torch.float32, device=device)
    hist = torch.tensor(np.array([b["history_feats"] for b in batch]), dtype=torch.float32, device=device)
    glob = torch.tensor(np.array([b["global_feats"] for b in batch]), dtype=torch.float32, device=device)
    mask = torch.tensor(np.array([b["action_mask"] for b in batch]), dtype=torch.float32, device=device)
    action = torch.tensor(np.array([b["action"] for b in batch]), dtype=torch.long, device=device)
    policy_target = None
    policy_target_mask = None
    if any("policy_target" in b for b in batch):
        policy_target = torch.tensor(
            np.array([b.get("policy_target", np.zeros(mask.shape[-1], dtype=np.float32)) for b in batch]),
            dtype=torch.float32,
            device=device,
        )
        policy_target_mask = torch.tensor(
            np.array([1.0 if "policy_target" in b else 0.0 for b in batch]),
            dtype=torch.float32,
            device=device,
        )
    policy_advantage = torch.tensor(
        np.array([b.get("policy_advantage", 1.0) for b in batch]),
        dtype=torch.float32,
        device=device,
    )
    policy_gradient_mask = torch.tensor(
        np.array([1.0 if b.get("policy_gradient", False) else 0.0 for b in batch]),
        dtype=torch.float32,
        device=device,
    )
    value = torch.tensor(np.array([b["value_target"] for b in batch]), dtype=torch.float32, device=device)
    value_sample_weight = torch.tensor(
        np.array([b.get("value_sample_weight", 1.0) for b in batch]),
        dtype=torch.float32,
        device=device,
    )
    btarget = torch.tensor(np.array([b["belief_target"] for b in batch]), dtype=torch.long, device=device)
    bmask = torch.tensor(np.array([b["belief_mask"] for b in batch]), dtype=torch.float32, device=device)
    af = action_features_torch(device).unsqueeze(0).expand(card.shape[0], -1, -1)
    return (
        card,
        hist,
        glob,
        mask,
        action,
        policy_target,
        policy_target_mask,
        policy_advantage,
        policy_gradient_mask,
        value,
        value_sample_weight,
        btarget,
        bmask,
        af,
    )


def _five_card_margin_loss(logits, mask, glob, af, margin):
    control = glob[:, 5] > 0.5
    five = (af[:, :, 55] > 0.5) & (mask > 0.5)
    non_five_play = (af[:, :, 53] + af[:, :, 54] > 0.5) & (mask > 0.5)
    eligible = control & five.any(dim=-1) & non_five_play.any(dim=-1)
    if not bool(eligible.any()):
        return logits.new_zeros(())

    five_scores = logits.masked_fill(~five, -1e9).max(dim=-1).values
    non_five_scores = logits.masked_fill(~non_five_play, -1e9).max(dim=-1).values
    return F.relu(non_five_scores[eligible] - five_scores[eligible] + float(margin)).mean()


def train_step(
    model,
    optimizer,
    batch,
    device,
    value_weight,
    belief_weight,
    policy_weight=1.0,
    q_value_weight=0.0,
    five_card_margin_weight=0.0,
    five_card_margin=0.5,
):
    (
        card,
        hist,
        glob,
        mask,
        action,
        policy_target,
        policy_target_mask,
        policy_advantage,
        policy_gradient_mask,
        value,
        value_sample_weight,
        btarget,
        bmask,
        af,
    ) = batch_to_tensors(batch, device)
    logits, pred_value, belief_logits = model(card, hist, glob, af, mask)
    if policy_target is None:
        policy_loss_all = F.cross_entropy(logits, action, reduction="none")
        if bool((policy_gradient_mask > 0.5).any()):
            pg_advantage = (value - pred_value.detach()).clamp(-1.0, 1.0)
            effective_advantage = torch.where(
                policy_gradient_mask > 0.5,
                pg_advantage,
                policy_advantage,
            )
        else:
            effective_advantage = policy_advantage
        policy_loss = (policy_loss_all * effective_advantage).mean()
    else:
        logp = F.log_softmax(logits, dim=-1)
        soft_loss = -(policy_target * logp).sum(dim=-1)
        hard_loss_all = F.cross_entropy(logits, action, reduction="none")
        pg_advantage = (value - pred_value.detach()).clamp(-1.0, 1.0)
        effective_advantage = torch.where(
            policy_gradient_mask > 0.5,
            pg_advantage,
            policy_advantage,
        )
        hard_loss = hard_loss_all * effective_advantage
        policy_loss = torch.where(policy_target_mask > 0.5, soft_loss, hard_loss).mean()
    value_error = (pred_value - value).pow(2)
    value_loss = (value_error * value_sample_weight).sum() / (value_sample_weight.sum() + 1e-6)
    q_value_loss = logits.new_zeros(())
    if q_value_weight > 0 and any("q_target" in b for b in batch):
        q_target = torch.tensor(
            np.array([b.get("q_target", np.zeros(mask.shape[-1], dtype=np.float32)) for b in batch]),
            dtype=torch.float32,
            device=device,
        )
        q_target_mask = torch.tensor(
            np.array([b.get("q_target_mask", np.zeros(mask.shape[-1], dtype=np.float32)) for b in batch]),
            dtype=torch.float32,
            device=device,
        )
        q_pred = model.action_values(card, hist, glob, af, mask)
        q_error = (q_pred - q_target).pow(2)
        q_value_loss = (q_error * q_target_mask).sum() / (q_target_mask.sum() + 1e-6)
    belief_loss_all = F.cross_entropy(
        belief_logits.reshape(-1, 3),
        btarget.reshape(-1),
        ignore_index=-1,
        reduction="none",
    ).reshape_as(bmask)
    belief_loss = (belief_loss_all * bmask).sum() / (bmask.sum() + 1e-6)
    five_margin_loss = _five_card_margin_loss(logits, mask, glob, af, five_card_margin)
    loss = (
        float(policy_weight) * policy_loss
        + value_weight * value_loss
        + float(q_value_weight) * q_value_loss
        + belief_weight * belief_loss
        + float(five_card_margin_weight) * five_margin_loss
    )
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    with torch.no_grad():
        pred = torch.argmax(logits, dim=-1)
        top1 = (pred == action).float().mean().item()
        mae = torch.mean(torch.abs(pred_value - value)).item()
    return {
        "loss": float(loss.item()),
        "policy_loss": float(policy_loss.item()),
        "policy_weight": float(policy_weight),
        "value_loss": float(value_loss.item()),
        "q_value_loss": float(q_value_loss.item()),
        "q_value_weight": float(q_value_weight),
        "belief_loss": float(belief_loss.item()),
        "five_card_margin_loss": float(five_margin_loss.item()),
        "policy_top1": float(top1),
        "pg_samples": float(policy_gradient_mask.sum().item()),
        "value_mae": float(mae),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--updates-per-episode", type=int, default=2)
    parser.add_argument("--buffer-capacity", type=int, default=200000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--value-scale", type=float, default=15.0)
    parser.add_argument("--policy-weight", type=float, default=1.0)
    parser.add_argument("--value-weight", type=float, default=0.5)
    parser.add_argument("--belief-weight", type=float, default=0.25)
    parser.add_argument("--model-selfplay-after", type=int, default=50)
    parser.add_argument("--model-selfplay-frac", type=float, default=0.25)
    parser.add_argument("--max-turns", type=int, default=300)
    parser.add_argument("--save", default="ML_AB/models/big2_transformer_current.pt")
    parser.add_argument("--metrics", default="ML_AB/runs/train_metrics.jsonl")
    parser.add_argument("--init", default="")
    args = parser.parse_args()

    configure_torch_threads()
    set_seed(args.seed)
    device = device_from_arg(args.device)
    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    os.makedirs(os.path.dirname(args.metrics), exist_ok=True)

    config = vars(args).copy()
    config["history_len"] = HISTORY_LEN
    model = Big2TransformerNet(
        d_model=args.d_model,
        nhead=args.heads,
        num_layers=args.layers,
        dropout=args.dropout,
        max_history_len=HISTORY_LEN,
    ).to(device)
    if args.init:
        payload = torch.load(args.init, map_location=device)
        state = payload["model_state"] if "model_state" in payload else payload
        load_compatible_state(model, state)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    replay = Replay(args.buffer_capacity)
    metrics_file = open(args.metrics, "a", buffering=1)
    t0 = time.time()

    for ep in range(1, args.episodes + 1):
        model_agent = ModelAgent(model, device=device, temperature=0.2)
        if ep >= args.model_selfplay_after and np.random.random() < args.model_selfplay_frac:
            mix = ("model", "heuristic", "random")
        else:
            mix = ("heuristic", "random")
        model.eval()
        samples, rewards = collect_episode(
            policy_mix=mix,
            model_agent=model_agent,
            value_scale=args.value_scale,
            max_turns=args.max_turns,
        )
        replay.add_many(samples)

        last = {}
        if len(replay) >= args.batch_size:
            model.train()
            for _ in range(args.updates_per_episode):
                batch = replay.sample(args.batch_size)
                last = train_step(
                    model,
                    optimizer,
                    batch,
                    device,
                    policy_weight=args.policy_weight,
                    value_weight=args.value_weight,
                    belief_weight=args.belief_weight,
                    five_card_margin_weight=0.0,
                )

        row = {
            "episode": ep,
            "buffer": len(replay),
            "samples": len(samples),
            "reward_mean": float(np.mean(rewards)),
            "reward_p1": float(rewards[0]),
            "elapsed_sec": round(time.time() - t0, 3),
            **last,
        }
        metrics_file.write(json.dumps(row, sort_keys=True) + "\n")
        if ep == 1 or ep % 10 == 0:
            print(json.dumps(row, sort_keys=True))

    metrics = {"episodes": args.episodes, "last": row}
    torch.save(checkpoint_payload(model, config, metrics), args.save)
    print(f"saved checkpoint: {args.save}")


if __name__ == "__main__":
    main()
