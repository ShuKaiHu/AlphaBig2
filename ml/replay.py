import argparse
import numpy as np

import big2Game
import enumerateOptions
import gameLogic


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


def choose_random_action(game):
    available = game.returnAvailableActions()
    valid = np.flatnonzero(available == 1)
    if valid.size == 0:
        return enumerateOptions.passInd
    non_pass = valid[valid != enumerateOptions.passInd]
    if non_pass.size > 0:
        return int(np.random.choice(non_pass))
    return int(np.random.choice(valid))


def replay_random(seed=None, max_turns=500):
    if seed is not None:
        np.random.seed(seed)
    game = big2Game.big2Game()

    turn = 0
    while not game.gameOver and turn < max_turns:
        print(f"\nTurn {turn + 1} (P{game.playersGo} to act)")
        print_hands(game)

        action = choose_random_action(game)
        player = game.playersGo
        if action == enumerateOptions.passInd:
            print(f"P{player}: PASS")
            game.updateGame(-1)
        else:
            opt, nC = enumerateOptions.getOptionNC(action)
            if nC == 1:
                hand = np.array([game.currentHands[player][opt]])
            elif nC == 2:
                hand = game.currentHands[player][
                    enumerateOptions.inverseTwoCardIndices[opt]
                ]
            else:
                hand = game.currentHands[player][
                    enumerateOptions.inverseFiveCardIndices[opt]
                ]
            prev = None
            if game.goIndex > 1:
                prev = game.handsPlayed[game.goIndex - 1].hand
            override = ""
            if prev is not None:
                if hand.size != prev.size and (
                    gameLogic.isFourOfAKind(hand) or gameLogic.isStraightFlush(hand)
                ):
                    override = " (override)"
            print(f"P{player}: {hand_to_str(hand)} [{hand_type(hand)}]{override}")
            game.updateGame(opt, nC)
        turn += 1

    print("\nFinal scores:")
    for i, score in enumerate(game.rewards, start=1):
        print(f"P{i}: {int(score)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-turns", type=int, default=500)
    replay_random(**vars(parser.parse_args()))
