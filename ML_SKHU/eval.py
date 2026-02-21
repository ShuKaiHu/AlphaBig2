import argparse
import numpy as np
import torch
import enumerateOptions
import big2Game

from ML_SKHU.belief import BeliefModel
from ML_SKHU.policy_value import PolicyValueModel
from ML_SKHU.features import belief_input_dim, encode_belief_input, encode_full_info_with_belief, belief_targets
from ML_SKHU.mcts import MCTS
from ML_SKHU.selfplay import make_policy_value_fn


def _random_action(game):
    avail = game.returnAvailableActions()
    valid = np.flatnonzero(avail == 1)
    if valid.size == 0:
        return enumerateOptions.passInd
    return int(np.random.choice(valid))

def _heuristic_action(game):
    avail = game.returnAvailableActions()
    valid = np.flatnonzero(avail == 1)
    if valid.size == 0:
        return enumerateOptions.passInd
    non_pass = valid[valid != enumerateOptions.passInd]
    if non_pass.size > 0:
        return int(np.min(non_pass))
    return enumerateOptions.passInd


def _apply_action(game, action):
    if action == enumerateOptions.passInd:
        game.updateGame(-1)
        return
    opt, n_cards = enumerateOptions.getOptionNC(action)
    game.updateGame(opt, n_cards)


def _policy_only_action(model, game, device="cpu"):
    avail = game.returnAvailableActions()
    valid = np.flatnonzero(avail == 1)
    if valid.size == 0:
        return enumerateOptions.passInd
    belief_in = encode_belief_input(game, game.playersGo)
    action_mask = np.isfinite(
        big2Game.convertAvailableActions(avail.astype(np.float32))
    ).astype(np.float32)
    with torch.no_grad():
        p_in = torch.from_numpy(belief_in).float().unsqueeze(0).to(device)
        mask_t = torch.from_numpy(action_mask).float().unsqueeze(0).to(device)
        logits, _ = model(p_in, mask_t)
        action = int(torch.argmax(logits, dim=-1).item())
    return action


def _load_model(model, path):
    if not path:
        return model, False
    try:
        state = torch.load(path, map_location="cpu")
        if isinstance(state, dict):
            model.load_state_dict(state)
            return model, True
    except Exception:
        raise
    scripted = torch.jit.load(path, map_location="cpu")
    return scripted, True


def _belief_accuracy(belief_model, game, player, device="cpu", margin=0.1):
    if belief_model is None:
        return None, None, 0, 0
    belief_in = encode_belief_input(game, player)
    targets, mask = belief_targets(game, player)
    with torch.no_grad():
        b_in = torch.from_numpy(belief_in).float().unsqueeze(0).to(device)
        logits = belief_model(b_in)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    preds = np.argmax(probs, axis=-1)
    valid = mask > 0
    if valid.sum() == 0:
        return None, None, 0, 0
    top2 = np.partition(probs, -2, axis=-1)[:, -2:]
    confidence = top2[:, 1] - top2[:, 0]
    known_mask = valid & (confidence >= margin)
    if known_mask.sum() == 0:
        return None, 0.0, 0, int(valid.sum())
    acc = (preds[known_mask] == targets[known_mask]).mean()
    coverage = float(known_mask.sum() / valid.sum())
    return float(acc), coverage, int(known_mask.sum()), int(valid.sum())


def _phase(turn_idx, total_turns):
    if total_turns <= 0:
        return "mid"
    third = total_turns // 3
    if turn_idx < third:
        return "early"
    if turn_idx < 2 * third:
        return "mid"
    return "late"


def play_game(
    belief_model,
    policy_value_model,
    policy_only_model,
    n_simulations,
    opponent_policy,
    device="cpu",
    margin=0.5,
):
    game = big2Game.big2Game()
    use_mcts = n_simulations and n_simulations > 0 and policy_value_model is not None
    if use_mcts:
        policy_value_fn = make_policy_value_fn(belief_model, policy_value_model, device=device)
        mcts = MCTS(policy_value_fn, n_simulations=n_simulations)
    belief_accs = []
    belief_coverages = []
    known_counts = []
    valid_counts = []
    per_turn = []
    turn_idx = 0

    while not game.gameOver:
        player = game.playersGo
        if player == 1:
            if use_mcts:
                action, _ = mcts.select_action(game, player, temperature=0.0)
            elif policy_only_model is not None:
                action = _policy_only_action(policy_only_model, game, device=device)
            else:
                action = _heuristic_action(game)
        else:
            if opponent_policy == "heuristic":
                action = _heuristic_action(game)
            else:
                action = _random_action(game)
        acc = coverage = None
        known_count = valid_count = 0
        if player == 1:
            acc, coverage, known_count, valid_count = _belief_accuracy(
                belief_model, game, player, device=device, margin=margin
            )
            if acc is not None:
                belief_accs.append(acc)
            if coverage is not None:
                belief_coverages.append(coverage)
            if valid_count > 0:
                known_counts.append(known_count)
                valid_counts.append(valid_count)
        per_turn.append((turn_idx, acc, coverage, known_count, valid_count))
        _apply_action(game, action)
        turn_idx += 1

    rewards = [float(r) for r in game.rewards]
    wins = [1 if r > 0 else 0 for r in rewards]
    phase_stats = {
        "early": {"acc_sum": 0.0, "acc_n": 0, "cov_sum": 0.0, "cov_n": 0, "known": 0, "valid": 0},
        "mid": {"acc_sum": 0.0, "acc_n": 0, "cov_sum": 0.0, "cov_n": 0, "known": 0, "valid": 0},
        "late": {"acc_sum": 0.0, "acc_n": 0, "cov_sum": 0.0, "cov_n": 0, "known": 0, "valid": 0},
    }
    total_turns = turn_idx
    for t, acc, cov, kcnt, vcnt in per_turn:
        phase = _phase(t, total_turns)
        if acc is not None:
            phase_stats[phase]["acc_sum"] += float(acc)
            phase_stats[phase]["acc_n"] += 1
        if cov is not None:
            phase_stats[phase]["cov_sum"] += float(cov)
            phase_stats[phase]["cov_n"] += 1
        phase_stats[phase]["known"] += int(kcnt)
        phase_stats[phase]["valid"] += int(vcnt)

    return wins, rewards, belief_accs, belief_coverages, known_counts, valid_counts, phase_stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--simulations", type=int, default=25)
    parser.add_argument("--belief-ckpt", type=str, default="")
    parser.add_argument("--policy-ckpt", type=str, default="")
    parser.add_argument("--policy-only-ckpt", type=str, default="")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--opponent", type=str, default="random", choices=["random", "heuristic"])
    parser.add_argument("--belief-margin", type=float, default=0.5)
    args = parser.parse_args()

    b_input_dim = belief_input_dim()
    dummy = big2Game.big2Game()
    dummy_belief = np.zeros((52, 3), dtype=np.float32)
    p_input_dim = int(encode_full_info_with_belief(dummy, 1, dummy_belief).shape[0])

    belief_model = BeliefModel(b_input_dim).to(args.device)
    policy_value_model = PolicyValueModel(p_input_dim).to(args.device)
    policy_only_model = None

    belief_model, loaded_belief = _load_model(belief_model, args.belief_ckpt)
    policy_value_model, loaded_policy = _load_model(policy_value_model, args.policy_ckpt)
    if args.policy_only_ckpt:
        policy_only_model = PolicyValueModel(b_input_dim, hidden_dim=256).to(args.device)
        policy_only_model.load_state_dict(torch.load(args.policy_only_ckpt, map_location=args.device))
        policy_only_model.eval()
    if not (loaded_belief or loaded_policy):
        print("warning: no checkpoints provided; models are randomly initialized.")

    wins = [0, 0, 0, 0]
    rewards = [[], [], [], []]
    belief_accs = []
    belief_coverages = []
    total_known = 0
    total_valid = 0
    for _ in range(args.games):
        win, reward, accs, coverages, kcounts, vcounts, _ = play_game(
            belief_model,
            policy_value_model,
            policy_only_model,
            args.simulations,
            args.opponent,
            device=args.device,
            margin=args.belief_margin,
        )
        for i in range(4):
            wins[i] += int(win[i])
            rewards[i].append(reward[i])
        belief_accs.extend(accs)
        belief_coverages.extend(coverages)
        total_known += int(np.sum(kcounts)) if kcounts else 0
        total_valid += int(np.sum(vcounts)) if vcounts else 0

    win_rates = [w / args.games for w in wins]
    avg_rewards = [float(np.mean(r)) if r else 0.0 for r in rewards]
    avg_belief = float(np.mean(belief_accs)) if belief_accs else None
    avg_coverage = float(np.mean(belief_coverages)) if belief_coverages else None

    print(f"games: {args.games}")
    for i in range(4):
        print(f"player{i+1} win_rate: {win_rates[i]:.2f} avg_reward: {avg_rewards[i]:.2f}")
    if avg_belief is not None:
        print(f"belief_top1_acc_known (avg): {avg_belief:.3f}")
    if avg_coverage is not None:
        print(f"belief_known_coverage (avg): {avg_coverage:.3f}")
    if total_valid > 0:
        print(f"belief_known_count: {total_known}/{total_valid}")


if __name__ == "__main__":
    main()
