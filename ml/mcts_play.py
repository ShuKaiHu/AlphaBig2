import argparse
import numpy as np

import big2Game
import enumerateOptions
import gameLogic

import torch

from ml.mcts import mcts_search, decode_action
from ml.train_belief import BeliefNet
from ml.policy import PolicyNet


def card_to_str(card_id):
    value = int(np.ceil(card_id / 4))
    if value <= 7:
        rank = str(value + 2)
    elif value == 8:
        rank = "10"
    elif value == 9:
        rank = "J"
    elif value == 10:
        rank = "Q"
    elif value == 11:
        rank = "K"
    elif value == 12:
        rank = "A"
    else:
        rank = "2"
    suit_val = card_id % 4
    if suit_val == 0:
        suit_val = 4
    suit = {1: "C", 2: "D", 3: "H", 4: "S"}[suit_val]
    return f"{rank}{suit}"


def hand_to_str(hand):
    return " ".join(card_to_str(c) for c in sorted(hand))


def hand_type(hand):
    if hand.size == 1:
        return "single"
    if hand.size == 2:
        return "pair"
    if hand.size == 5:
        if gameLogic.isStraightFlush(hand):
            return "straight_flush"
        if gameLogic.isFourOfAKind(hand):
            return "four_kind"
        if gameLogic.isFullHouse(hand)[0]:
            return "full_house"
        if gameLogic.isStraight(hand):
            return "straight"
    return "unknown"


def print_hands(game):
    for i in range(1, 5):
        cards = hand_to_str(game.currentHands[i])
        print(f"P{i} hand ({len(game.currentHands[i])}): {cards}")


def play(args):
    game = big2Game.big2Game()
    belief_model = None
    if args.belief_model:
        data = np.load(args.belief_data)
        belief_model = BeliefNet(data["x"].shape[1])
        payload = torch.load(args.belief_model, map_location="cpu")
        belief_model.load_state_dict(payload["model_state"])
        belief_model.eval()
    rollout_policy = None
    if args.rollout_model:
        rollout_policy = PolicyNet()
        payload = torch.load(args.rollout_model, map_location="cpu")
        rollout_policy.load_state_dict(payload["model_state"])
        rollout_policy.eval()
    turn = 0
    while not game.gameOver and turn < args.max_turns:
        print(f"\nTurn {turn + 1} (P{game.playersGo} to act)")
        print_hands(game)

        player = game.playersGo
        if player == args.mcts_player:
            action = mcts_search(
                game,
                simulations=args.simulations,
                max_rollout_steps=args.rollout_steps,
                belief_model=belief_model,
                temperature=args.temperature,
                risk_aversion=args.risk_aversion,
                rollout_policy=rollout_policy,
                policy_model=rollout_policy,
            )
        else:
            action = enumerateOptions.passInd
            valid = np.flatnonzero(game.returnAvailableActions() == 1)
            non_pass = valid[valid != enumerateOptions.passInd]
            if non_pass.size > 0:
                action = int(np.random.choice(non_pass))
            elif valid.size > 0:
                action = int(np.random.choice(valid))

        if action == enumerateOptions.passInd:
            print(f"P{player}: PASS")
            game.updateGame(-1)
        else:
            opt, nC, hand = decode_action(game, action)
            print(f"P{player}: {hand_to_str(hand)} [{hand_type(hand)}]")
            game.updateGame(opt, nC)
        turn += 1

    print("\nFinal scores:")
    for i, score in enumerate(game.rewards, start=1):
        print(f"P{i}: {int(score)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mcts-player", type=int, default=1)
    parser.add_argument("--simulations", type=int, default=100)
    parser.add_argument("--rollout-steps", type=int, default=200)
    parser.add_argument("--max-turns", type=int, default=200)
    parser.add_argument("--belief-model", type=str, default="")
    parser.add_argument("--belief-data", type=str, default="ml/belief_dataset.npz")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--risk-aversion", type=float, default=0.0)
    parser.add_argument("--rollout-model", type=str, default="")
    play(parser.parse_args())
