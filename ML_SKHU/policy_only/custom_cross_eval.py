import argparse
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch

import big2Game
import enumerateOptions

from ML_SKHU.features import (
    encode_belief_input,
    encode_full_info_input,
    encode_full_info_input_known,
)
from ML_SKHU.policy_value import PolicyValueModel


def heuristic_action(game):
    avail = game.returnAvailableActions()
    valid = np.flatnonzero(avail == 1)
    if valid.size == 0:
        return enumerateOptions.passInd
    non_pass = valid[valid != enumerateOptions.passInd]
    if non_pass.size > 0:
        return int(np.min(non_pass))
    return enumerateOptions.passInd


def random_action(game):
    avail = game.returnAvailableActions()
    valid = np.flatnonzero(avail == 1)
    if valid.size == 0:
        return enumerateOptions.passInd
    return int(np.random.choice(valid))


def _get_encoder(mode):
    if mode == "belief":
        return encode_belief_input
    if mode == "full_info":
        return encode_full_info_input
    if mode == "full_info_known":
        return encode_full_info_input_known
    raise ValueError(f"unknown encoder mode: {mode}")


@dataclass
class ModelSpec:
    name: str
    model: Optional[PolicyValueModel]
    encoder: Optional[Callable]
    kind: str


def load_model(path, encoder, device="cpu"):
    sample = encoder(big2Game.big2Game(), 1)
    model = PolicyValueModel(sample.shape[0], hidden_dim=256).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def action_from_model(model, encoder, game, device="cpu"):
    belief_in = encoder(game, game.playersGo)
    avail = game.returnAvailableActions()
    valid = np.flatnonzero(avail == 1)
    if valid.size == 0:
        return enumerateOptions.passInd
    avail = avail.astype(np.float32)
    mask = torch.tensor(
        np.isfinite(big2Game.convertAvailableActions(avail)).astype(np.float32)
    ).unsqueeze(0).to(device)
    with torch.no_grad():
        logits, _ = model(torch.from_numpy(belief_in).float().unsqueeze(0).to(device), mask)
    return int(torch.argmax(logits, dim=-1).item())


def run_game(models: list[ModelSpec], device="cpu"):
    game = big2Game.big2Game()
    while not game.gameOver:
        player = game.playersGo
        spec = models[player - 1]
        if spec.model is None:
            if spec.kind == "random":
                action = random_action(game)
            else:
                action = heuristic_action(game)
        else:
            action = action_from_model(spec.model, spec.encoder, game, device=device)
        if action == enumerateOptions.passInd:
            game.updateGame(-1)
        else:
            opt, n = enumerateOptions.getOptionNC(action)
            game.updateGame(opt, n)
    return [float(r) for r in game.rewards]


def format_table(table: np.ndarray, title: str, labels: list[str]):
    header = "opp\\player"
    col_header = " ".join(f"{label:>14}" for label in labels)
    lines = [title, f"{header:>12} {col_header}"]
    for row_idx, row_label in enumerate(labels):
        row_vals = [f"{table[row_idx, col_idx]:+14.2f}" for col_idx in range(table.shape[1])]
        lines.append(f"{row_label:>8} " + " ".join(row_vals))
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model-a", required=True, help="path to no-full-info model")
    parser.add_argument("--model-b", required=True, help="path to full-info model")
    parser.add_argument(
        "--info-mode",
        choices=["unknown", "full", "full_known"],
        default="unknown",
        help="force encoder for all models during eval",
    )
    args = parser.parse_args()

    device = args.device
    mode = args.info_mode

    spec_heur = ModelSpec("heur", None, None, "heur")
    spec_rand = ModelSpec("rand", None, None, "random")
    spec_a = ModelSpec(
        "no_fullinfo",
        load_model(args.model_a, _get_encoder("belief"), device=device),
        _get_encoder("belief"),
        "belief",
    )
    spec_b = ModelSpec(
        "fullinfo",
        load_model(args.model_b, _get_encoder("full_info_known"), device=device),
        _get_encoder("full_info_known"),
        "fullinfo",
    )

    if mode == "unknown":
        spec_a.encoder = _get_encoder("belief")
        spec_b.encoder = _get_encoder("full_info")
    elif mode == "full":
        spec_a.encoder = _get_encoder("belief")
        spec_b.encoder = _get_encoder("full_info")
    else:
        spec_a.encoder = _get_encoder("belief")
        spec_b.encoder = _get_encoder("full_info_known")

    specs = [spec_heur, spec_rand, spec_a, spec_b]
    labels = [s.name for s in specs]
    total = len(specs)

    reward_table = np.zeros((total, total), dtype=np.float32)

    for opp_idx, opp_spec in enumerate(specs):
        opp_models = [opp_spec] * 3
        for player_idx, player_spec in enumerate(specs):
            models_list = [player_spec] + opp_models
            rewards = []
            for _ in range(args.games):
                rewards.append(run_game(models_list, device=device)[0])
            reward_table[opp_idx, player_idx] = float(np.mean(rewards))

    print("\n".join(format_table(reward_table, "player1 reward only:", labels)))


if __name__ == "__main__":
    main()
