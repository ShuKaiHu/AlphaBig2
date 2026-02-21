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


def select_action(game, actor):
    if actor == "heuristic":
        return _min_play_action(game)
    if actor == "random":
        return random_action(game)
    raise ValueError(f"unknown actor {actor}")


def eval_model(model, games=200, actor="heuristic", device="cpu", reward_gamma=0.99):
    preds = []
    targets = []

    for _ in range(games):
        game = big2Game.big2Game()
        p1_action_index = 0
        decision_times = []
        pred_buffer = []

        while not game.gameOver:
            player = game.playersGo
            if player == 1:
                obs = encode_value_input_fullinfo(game)
                mask = np.isfinite(
                    big2Game.convertAvailableActions(
                        game.returnAvailableActions().astype(np.float32)
                    )
                ).astype(np.float32)
                with torch.no_grad():
                    _, value = model(
                        torch.from_numpy(obs).float().unsqueeze(0).to(device),
                        torch.from_numpy(mask).float().unsqueeze(0).to(device),
                    )
                pred_buffer.append(float(value.item()))
                decision_times.append(p1_action_index)

            action = select_action(game, actor)
            opt, n = enumerateOptions.getOptionNC(action)
            game.updateGame(opt, n)
            if player == 1 and action != enumerateOptions.passInd:
                p1_action_index += 1

        reward = float(game.rewards[0])
        last_index = max(0, p1_action_index - 1)
        for i, pred in enumerate(pred_buffer):
            steps_behind = last_index - decision_times[i]
            decay = reward_gamma ** steps_behind
            target = reward * decay
            preds.append(pred)
            targets.append(target)

    preds = np.array(preds, dtype=np.float32)
    targets = np.array(targets, dtype=np.float32)
    mse = float(np.mean((preds - targets) ** 2))
    mae = float(np.mean(np.abs(preds - targets)))
    rmse = float(np.sqrt(mse))
    if preds.size > 1:
        corr = float(np.corrcoef(preds, targets)[0, 1])
    else:
        corr = 0.0
    return mse, mae, rmse, corr, preds, targets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--actor", choices=["heuristic", "random"], default="heuristic")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--reward-gamma", type=float, default=0.99)
    args = parser.parse_args()

    model = load_model(args.ckpt, device=args.device)
    mse, mae, rmse, corr, preds, targets = eval_model(
        model,
        games=args.games,
        actor=args.actor,
        device=args.device,
        reward_gamma=args.reward_gamma,
    )

    print(f"samples: {len(preds)}")
    print(f"mse: {mse:.4f}")
    print(f"mae: {mae:.4f}")
    print(f"rmse: {rmse:.4f}")
    print(f"corr: {corr:.4f}")


if __name__ == "__main__":
    main()
