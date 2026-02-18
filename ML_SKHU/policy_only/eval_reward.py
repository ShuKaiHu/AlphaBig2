import argparse
import numpy as np
import torch

import big2Game
import enumerateOptions

from ML_SKHU.policy_value import PolicyValueModel
from ML_SKHU.features import (
    encode_belief_input,
    encode_full_info_input,
    encode_full_info_input_known,
)


def heuristic_action(game):
    avail = game.returnAvailableActions()
    valid = np.flatnonzero(avail == 1)
    if valid.size == 0:
        return enumerateOptions.passInd
    non_pass = valid[valid != enumerateOptions.passInd]
    return int(np.min(non_pass)) if non_pass.size > 0 else enumerateOptions.passInd


def play_game(
    model,
    device="cpu",
    opp_actor="heuristic",
    opp_model=None,
    full_info=False,
    full_info_known=False,
):
    model.eval()
    if full_info_known:
        encoder = encode_full_info_input_known
    else:
        encoder = encode_full_info_input if full_info else encode_belief_input
    game = big2Game.big2Game()
    while not game.gameOver:
        player = game.playersGo
        if player == 1:
            belief_in = encoder(game, player)
            mask = torch.tensor(
                np.isfinite(
                    big2Game.convertAvailableActions(
                        game.returnAvailableActions().astype(np.float32)
                    )
                ).astype(np.float32)
            ).unsqueeze(0).to(device)
            with torch.no_grad():
                logits, _ = model(
                    torch.from_numpy(belief_in).float().unsqueeze(0).to(device), mask
                )
            action = int(torch.argmax(logits, dim=-1).item())
        else:
            if opp_actor == "heuristic":
                action = heuristic_action(game)
            elif opp_actor == "random":
                avail = game.returnAvailableActions()
                valid = np.flatnonzero(avail == 1)
                action = int(np.random.choice(valid)) if valid.size > 0 else enumerateOptions.passInd
            elif opp_actor == "model":
                belief_in = encoder(game, player)
                mask = torch.tensor(
                    np.isfinite(
                        big2Game.convertAvailableActions(
                            game.returnAvailableActions().astype(np.float32)
                        )
                    ).astype(np.float32)
                ).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits, _ = opp_model(
                        torch.from_numpy(belief_in).float().unsqueeze(0).to(device), mask
                    )
                action = int(torch.argmax(logits, dim=-1).item())
            else:
                raise ValueError(f"unsupported opp_actor {opp_actor}")
        if action == enumerateOptions.passInd:
            game.updateGame(-1)
        else:
            opt, n = enumerateOptions.getOptionNC(action)
            game.updateGame(opt, n)

    return float(game.rewards[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-ckpt", required=True)
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--opponent", choices=["heuristic", "random", "model"], default="heuristic")
    parser.add_argument("--opponent-model", default=None)
    parser.add_argument("--full-info", action="store_true")
    parser.add_argument("--full-info-known", action="store_true")
    args = parser.parse_args()

    if args.full_info_known:
        encoder = encode_full_info_input_known
    else:
        encoder = encode_full_info_input if args.full_info else encode_belief_input
    sample_input = encoder(big2Game.big2Game(), 1)
    model = PolicyValueModel(sample_input.shape[0], hidden_dim=256).to(args.device)
    model.load_state_dict(torch.load(args.policy_ckpt, map_location=args.device))

    opp_model = None
    if args.opponent == "model":
        if not args.opponent_model:
            raise ValueError("opponent-model is required when opponent=model")
        opp_model = PolicyValueModel(sample_input.shape[0], hidden_dim=256).to(args.device)
        opp_model.load_state_dict(torch.load(args.opponent_model, map_location=args.device))
        opp_model.eval()

    rewards = []
    for _ in range(args.games):
            rewards.append(
                play_game(
                    model,
                    device=args.device,
                    opp_actor=args.opponent,
                    opp_model=opp_model,
                    full_info=args.full_info,
                    full_info_known=args.full_info_known,
                )
            )

    avg_reward = np.mean(rewards)
    print(f"{avg_reward:.6f}")


if __name__ == "__main__":
    main()
