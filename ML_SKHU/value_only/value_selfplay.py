import argparse
import numpy as np
import torch

import big2Game
import enumerateOptions

from ML_SKHU.features import encode_value_input_fullinfo
from ML_SKHU.policy_value import PolicyValueModel
from ML_SKHU.selfplay import _min_play_action


def load_model(path, device="cpu"):
    sample = encode_value_input_fullinfo(big2Game.big2Game())
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
        if model is None:
            return random_action(game)
        obs = encode_value_input_fullinfo(game)
        mask = np.isfinite(
            big2Game.convertAvailableActions(
                game.returnAvailableActions().astype(np.float32)
            )
        ).astype(np.float32)
        with torch.no_grad():
            logits, _ = model(
                torch.from_numpy(obs).float().unsqueeze(0).to(device),
                torch.from_numpy(mask).float().unsqueeze(0).to(device),
            )
        return int(torch.argmax(logits, dim=-1).item())
    raise ValueError(f"unknown actor {actor}")


def collect_data(games, actor="heuristic", model=None, device="cpu", reward_gamma=0.99):
    obs = []
    masks = []
    values = []

    for _ in range(games):
        game = big2Game.big2Game()
        player1_steps = 0
        p1_action_index = 0
        decision_times = []

        while not game.gameOver:
            player = game.playersGo
            action = select_action(game, actor, model, device)

            if player == 1:
                obs.append(encode_value_input_fullinfo(game))
                masks.append(
                    np.isfinite(
                        big2Game.convertAvailableActions(
                            game.returnAvailableActions().astype(np.float32)
                        )
                    ).astype(np.float32)
                )
                player1_steps += 1
                decision_times.append(p1_action_index)

            opt, n = enumerateOptions.getOptionNC(action)
            game.updateGame(opt, n)
            if player == 1 and action != enumerateOptions.passInd:
                p1_action_index += 1

        reward = float(game.rewards[0])
        last_index = max(0, p1_action_index - 1)
        for i in range(player1_steps):
            steps_behind = last_index - decision_times[i]
            decay = reward_gamma ** steps_behind
            values.append(reward * decay)

    return (
        np.stack(obs),
        np.stack(masks),
        np.array(values, dtype=np.float32),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--actor", choices=["heuristic", "random", "model"], default="heuristic")
    parser.add_argument("--model", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--reward-gamma", type=float, default=0.99)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    model = None
    if args.actor == "model" and args.model:
        model = load_model(args.model, device=args.device)

    obs, masks, values = collect_data(
        args.games,
        actor=args.actor,
        model=model,
        device=args.device,
        reward_gamma=args.reward_gamma,
    )

    np.savez(args.out, obs=obs, mask=masks, values=values)
    print(f"saved {args.out} samples={obs.shape[0]}")


if __name__ == "__main__":
    main()
