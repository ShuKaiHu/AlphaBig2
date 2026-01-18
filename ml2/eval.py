import argparse
import numpy as np
import torch
import enumerateOptions
import big2Game

from ml2.belief import BeliefModel
from ml2.policy_value import PolicyValueModel
from ml2.features import belief_input_dim, encode_belief_input, encode_policy_input, belief_targets
from ml2.mcts import MCTS
from ml2.selfplay import make_policy_value_fn


def _random_action(game):
    avail = game.returnAvailableActions()
    valid = np.flatnonzero(avail == 1)
    if valid.size == 0:
        return enumerateOptions.passInd
    return int(np.random.choice(valid))


def _apply_action(game, action):
    if action == enumerateOptions.passInd:
        game.updateGame(-1)
        return
    opt, n_cards = enumerateOptions.getOptionNC(action)
    game.updateGame(opt, n_cards)


def _load_model(model, path):
    if not path:
        return False
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    return True


def _belief_accuracy(belief_model, game, player, device="cpu"):
    if belief_model is None:
        return None
    belief_in = encode_belief_input(game, player)
    targets, mask = belief_targets(game, player)
    with torch.no_grad():
        b_in = torch.from_numpy(belief_in).float().unsqueeze(0).to(device)
        logits = belief_model(b_in)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    preds = np.argmax(probs, axis=-1)
    valid = mask > 0
    if valid.sum() == 0:
        return None
    acc = (preds[valid] == targets[valid]).mean()
    return float(acc)


def play_game(belief_model, policy_value_model, n_simulations, device="cpu"):
    game = big2Game.big2Game()
    policy_value_fn = make_policy_value_fn(belief_model, policy_value_model, device=device)
    mcts = MCTS(policy_value_fn, n_simulations=n_simulations)
    belief_accs = []

    while not game.gameOver:
        player = game.playersGo
        if player == 1:
            action, _ = mcts.select_action(game, player, temperature=0.0)
        else:
            action = _random_action(game)
        acc = _belief_accuracy(belief_model, game, player, device=device)
        if acc is not None:
            belief_accs.append(acc)
        _apply_action(game, action)

    reward = float(game.rewards[0])
    win = reward > 0
    return win, reward, belief_accs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--simulations", type=int, default=25)
    parser.add_argument("--belief-ckpt", type=str, default="")
    parser.add_argument("--policy-ckpt", type=str, default="")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    b_input_dim = belief_input_dim()
    p_input_dim = b_input_dim + 52 * 3

    belief_model = BeliefModel(b_input_dim).to(args.device)
    policy_value_model = PolicyValueModel(p_input_dim).to(args.device)

    loaded_belief = _load_model(belief_model, args.belief_ckpt)
    loaded_policy = _load_model(policy_value_model, args.policy_ckpt)
    if not (loaded_belief or loaded_policy):
        print("warning: no checkpoints provided; models are randomly initialized.")

    wins = 0
    rewards = []
    belief_accs = []
    for _ in range(args.games):
        win, reward, accs = play_game(
            belief_model, policy_value_model, args.simulations, device=args.device
        )
        wins += int(win)
        rewards.append(reward)
        belief_accs.extend(accs)

    win_rate = wins / args.games
    avg_reward = float(np.mean(rewards)) if rewards else 0.0
    avg_belief = float(np.mean(belief_accs)) if belief_accs else None

    print(f"games: {args.games}")
    print(f"win_rate (player1 vs random): {win_rate:.2f}")
    print(f"avg_reward (player1): {avg_reward:.2f}")
    if avg_belief is not None:
        print(f"belief_top1_acc (avg): {avg_belief:.3f}")


if __name__ == "__main__":
    main()
