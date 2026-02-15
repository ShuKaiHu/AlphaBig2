import argparse
import numpy as np
import torch

import big2Game
import enumerateOptions

from ML_SKHU.features import encode_belief_input, belief_targets
from ML_SKHU.selfplay import _min_play_action
from ML_SKHU.policy_value import PolicyValueModel


def load_policy_model(path, device="cpu"):
    sample = encode_belief_input(big2Game.big2Game(), 1)
    model = PolicyValueModel(sample.shape[0], hidden_dim=256).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def random_action(game):
    avail = game.returnAvailableActions()
    valid = np.flatnonzero(avail == 1)
    if valid.size == 0:
        return enumerateOptions.passInd
    return int(np.random.choice(valid))


def select_action(game, actor, model, device):
    if actor == "heuristic":
        return _min_play_action(game)
    if actor == "random":
        return random_action(game)
    if actor == "model":
        belief_in = encode_belief_input(game, game.playersGo)
        mask = np.isfinite(
            big2Game.convertAvailableActions(
                game.returnAvailableActions().astype(np.float32)
            )
        ).astype(np.float32)
        mask_tensor = torch.tensor(mask).unsqueeze(0).to(device)
        with torch.no_grad():
            logits, _ = model(
                torch.from_numpy(belief_in).float().unsqueeze(0).to(device),
                mask_tensor,
            )
        return int(torch.argmax(logits, dim=-1).item())
    raise ValueError(f"unknown actor {actor}")


def collect_data(
    games,
    player_actor="heuristic",
    opp_actor="heuristic",
    player_model=None,
    opp_model=None,
    device="cpu",
):
    obs = []
    masks = []
    actions = []
    values = []
    for _ in range(games):
        game = big2Game.big2Game()
        while not game.gameOver:
            player = game.playersGo
            belief_in = encode_belief_input(game, player)
            mask = np.isfinite(
                big2Game.convertAvailableActions(
                    game.returnAvailableActions().astype(np.float32)
                )
            ).astype(np.float32)

            if player == 1:
                action = select_action(
                    game, player_actor, player_model, device=device
                )
            else:
                action = select_action(game, opp_actor, opp_model, device=device)

            opt, n = enumerateOptions.getOptionNC(action)
            game.updateGame(opt, n)

            obs.append(belief_in)
            masks.append(mask)
            actions.append(action)

        reward = game.rewards[0]
        for _ in range(len(obs) - len(values)):
            values.append(1.0 if reward > 0 else -1.0)

    return np.stack(obs), np.stack(masks), np.array(actions), np.array(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument(
        "--player-actor", choices=["heuristic", "random", "model"], default="heuristic"
    )
    parser.add_argument(
        "--opp-actor", choices=["heuristic", "random", "model"], default="heuristic"
    )
    parser.add_argument("--player-model", default=None)
    parser.add_argument("--opp-model", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    player_model = (
        load_policy_model(args.player_model, device=args.device)
        if args.player_actor == "model"
        else None
    )
    opp_model = (
        load_policy_model(args.opp_model, device=args.device)
        if args.opp_actor == "model"
        else None
    )

    obs, masks, actions, values = collect_data(
        args.games,
        player_actor=args.player_actor,
        opp_actor=args.opp_actor,
        player_model=player_model,
        opp_model=opp_model,
        device=args.device,
    )
    np.savez(args.out, obs=obs, mask=masks, actions=actions, values=values)
    print(f"saved {args.out} samples={obs.shape[0]}")


if __name__ == "__main__":
    main()
