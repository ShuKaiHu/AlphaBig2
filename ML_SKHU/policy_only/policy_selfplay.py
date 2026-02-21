import argparse
import numpy as np
import torch
from torch.distributions import Categorical

import big2Game
import enumerateOptions

from ML_SKHU.features import encode_belief_input, encode_full_info_input, belief_targets
from ML_SKHU.selfplay import _min_play_action
from ML_SKHU.policy_value import PolicyValueModel


def load_policy_model(path, device="cpu", full_info=False):
    encoder = encode_full_info_input if full_info else encode_belief_input
    sample = encoder(big2Game.big2Game(), 1)
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


def select_action(game, actor, model, device, epsilon=0.0, full_info=False):
    if epsilon > 0.0 and np.random.rand() < epsilon:
        return random_action(game)
    if actor == "heuristic":
        return _min_play_action(game)
    if actor == "random":
        return random_action(game)
    if actor == "model":
        if model is None:
            return random_action(game)
        avail = game.returnAvailableActions()
        valid = np.flatnonzero(avail == 1)
        if valid.size == 0:
            return enumerateOptions.passInd
        encoder = encode_full_info_input if full_info else encode_belief_input
        belief_in = encoder(game, game.playersGo)
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
        dist = Categorical(logits=logits)
        return int(dist.sample().item())
    raise ValueError(f"unknown actor {actor}")


def select_action_with_logprob(game, actor, model, device, epsilon=0.0, full_info=False):
    avail = game.returnAvailableActions()
    valid = np.flatnonzero(avail == 1)
    if valid.size == 0:
        return enumerateOptions.passInd, 0.0

    if epsilon > 0.0 and np.random.rand() < epsilon:
        action = int(np.random.choice(valid))
        logp = -np.log(valid.size)
        return action, float(logp)

    if actor == "heuristic":
        action = _min_play_action(game)
        logp = -np.log(valid.size)
        return action, float(logp)
    if actor == "random":
        action = int(np.random.choice(valid))
        logp = -np.log(valid.size)
        return action, float(logp)
    if actor == "model":
        if model is None:
            action = int(np.random.choice(valid))
            logp = -np.log(valid.size)
            return action, float(logp)
        encoder = encode_full_info_input if full_info else encode_belief_input
        belief_in = encoder(game, game.playersGo)
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
        dist = Categorical(logits=logits)
        action_t = dist.sample()
        logp = float(dist.log_prob(action_t).item())
        return int(action_t.item()), logp
    raise ValueError(f"unknown actor {actor}")


def collect_data(
    games,
    player_actor="heuristic",
    opp_actor="heuristic",
    player_model=None,
    opp_model=None,
    device="cpu",
    epsilon=0.0,
    reward_gamma=0.99,
    skip_forced_pass=True,
    full_info=False,
    reward_scale=1.0,
):
    obs = []
    masks = []
    actions = []
    old_logprobs = []
    values = []
    for _ in range(games):
        game = big2Game.big2Game()
        player1_steps = 0
        action_index = 0
        decision_times = []
        while not game.gameOver:
            player = game.playersGo
            encoder = encode_full_info_input if full_info else encode_belief_input
            belief_in = encoder(game, player)
            mask = np.isfinite(
                big2Game.convertAvailableActions(
                    game.returnAvailableActions().astype(np.float32)
                )
            ).astype(np.float32)
            avail = game.returnAvailableActions()
            valid = np.flatnonzero(avail == 1)
            only_pass = (
                valid.size == 1 and int(valid[0]) == enumerateOptions.passInd
            )

            if player == 1:
                action, logp = select_action_with_logprob(
                    game,
                    player_actor,
                    player_model,
                    device=device,
                    epsilon=epsilon,
                    full_info=full_info,
                )
            else:
                action = select_action(
                    game, opp_actor, opp_model, device=device, full_info=full_info
                )

            opt, n = enumerateOptions.getOptionNC(action)
            game.updateGame(opt, n)
            if action != enumerateOptions.passInd:
                action_index += 1

            if player == 1:
                if not (skip_forced_pass and only_pass):
                    obs.append(belief_in)
                    masks.append(mask)
                    actions.append(action)
                    old_logprobs.append(logp)
                    player1_steps += 1
                    decision_times.append(action_index)

        reward = game.rewards[0]
        last_index = max(0, action_index - 1)
        for i in range(player1_steps):
            steps_behind = last_index - decision_times[i]
            decay = reward_gamma ** steps_behind
            scaled_reward = float(reward) / reward_scale
            values.append(scaled_reward * decay)

    return (
        np.stack(obs),
        np.stack(masks),
        np.array(actions),
        np.array(old_logprobs, dtype=np.float32),
        np.array(values),
    )


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
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--reward-gamma", type=float, default=0.99)
    parser.add_argument("--skip-forced-pass", action="store_true")
    parser.add_argument("--full-info", action="store_true")
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    player_model = None
    if args.player_actor == "model" and args.player_model:
        player_model = load_policy_model(
            args.player_model, device=args.device, full_info=args.full_info
        )
    opp_model = None
    if args.opp_actor == "model" and args.opp_model:
        opp_model = load_policy_model(
            args.opp_model, device=args.device, full_info=args.full_info
        )

    obs, masks, actions, old_logprobs, values = collect_data(
        args.games,
        player_actor=args.player_actor,
        opp_actor=args.opp_actor,
        player_model=player_model,
        opp_model=opp_model,
        device=args.device,
        epsilon=args.epsilon,
        reward_gamma=args.reward_gamma,
        skip_forced_pass=args.skip_forced_pass,
        full_info=args.full_info,
        reward_scale=args.reward_scale,
    )
    np.savez(
        args.out,
        obs=obs,
        mask=masks,
        actions=actions,
        old_logprobs=old_logprobs,
        values=values,
    )
    print(f"saved {args.out} samples={obs.shape[0]}")


if __name__ == "__main__":
    main()
