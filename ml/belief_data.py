import argparse
import numpy as np
import torch

import big2Game
import enumerateOptions
import gameLogic
from ml.mcts import mcts_search
from ml.train_belief import BeliefNet


ACTION_TYPES = {
    "pass": 0,
    "single": 1,
    "pair": 2,
    "straight": 3,
    "full_house": 4,
    "four_kind": 5,
    "straight_flush": 6,
}


def _hand_type(hand):
    if hand is None or hand.size == 0:
        return "pass"
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
    return "pass"


def _action_vector(player_id, hand, pass_count, control_flag):
    vec = np.zeros((4 + 7 + 52 + 1 + 1,), dtype=np.float32)
    vec[player_id - 1] = 1.0
    hand_type = _hand_type(hand)
    vec[4 + ACTION_TYPES[hand_type]] = 1.0
    if hand is not None and hand.size > 0:
        for card in hand:
            vec[4 + 7 + (card - 1)] = 1.0
    vec[-2] = float(pass_count) / 3.0
    vec[-1] = 1.0 if control_flag else 0.0
    return vec


def _build_sample(game, history, history_len):
    # history: list of action vectors
    if len(history) < history_len:
        pad = [np.zeros_like(history[0]) for _ in range(history_len - len(history))]
        hist = pad + history
    else:
        hist = history[-history_len:]
    hist_flat = np.concatenate(hist, axis=0)

    own_hand = np.zeros((52,), dtype=np.float32)
    for card in game.currentHands[1]:
        own_hand[card - 1] = 1.0

    played = np.zeros((52,), dtype=np.float32)
    played_mask = np.sum(game.cardsPlayed, axis=0)
    played[played_mask > 0] = 1.0

    counts = np.array(
        [
            game.currentHands[1].size,
            game.currentHands[2].size,
            game.currentHands[3].size,
            game.currentHands[4].size,
        ],
        dtype=np.float32,
    )
    counts = counts / 13.0

    x = np.concatenate([hist_flat, own_hand, played, counts], axis=0)

    # Labels: for each unknown card, which opponent holds it (0,1,2).
    y = np.full((52,), -1, dtype=np.int64)
    mask = np.zeros((52,), dtype=np.float32)
    opp_hands = {2: game.currentHands[2], 3: game.currentHands[3], 4: game.currentHands[4]}
    for card in range(1, 53):
        if own_hand[card - 1] == 1.0 or played[card - 1] == 1.0:
            continue
        for opp_index, opp_id in enumerate([2, 3, 4]):
            if card in opp_hands[opp_id]:
                y[card - 1] = opp_index
                mask[card - 1] = 1.0
                break
    return x, y, mask


def _decode_action(game, action):
    player = game.playersGo
    opt, nC = enumerateOptions.getOptionNC(action)
    if nC == 1:
        hand = np.array([game.currentHands[player][opt]])
    elif nC == 2:
        hand = game.currentHands[player][enumerateOptions.inverseTwoCardIndices[opt]]
    else:
        hand = game.currentHands[player][enumerateOptions.inverseFiveCardIndices[opt]]
    return opt, nC, hand


def _choose_heuristic_action(game):
    available = game.returnAvailableActions()
    valid = np.flatnonzero(available == 1)
    if valid.size == 0:
        return -1
    non_pass = valid[valid != enumerateOptions.passInd]
    if non_pass.size == 0:
        return -1

    if game.control == 1:
        # Prefer 5-card, then pair, then single.
        for nC in (5, 2, 1):
            candidates = []
            for action in non_pass:
                _opt, a_nC = enumerateOptions.getOptionNC(int(action))
                if a_nC == nC:
                    candidates.append(int(action))
            if candidates:
                return min(candidates)
        return int(non_pass[0])

    # Not in control: try to follow current hand size first.
    prev_hand = game.handsPlayed[game.goIndex - 1].hand
    prev_size = len(prev_hand)
    candidates = []
    for action in non_pass:
        _opt, a_nC = enumerateOptions.getOptionNC(int(action))
        if a_nC == prev_size:
            candidates.append(int(action))
    if candidates:
        return min(candidates)

    # If no same-size option, prefer override (four-kind or straight-flush).
    override_candidates = []
    for action in non_pass:
        _opt, a_nC, hand = _decode_action(game, int(action))
        if a_nC == 5 and (gameLogic.isFourOfAKind(hand) or gameLogic.isStraightFlush(hand)):
            override_candidates.append(int(action))
    if override_candidates:
        return min(override_candidates)

    return -1


def _choose_random_action(game):
    return game.randomOption()


def generate_dataset(
    num_games,
    history_len,
    max_turns,
    seed=None,
    policy="heuristic",
    mcts_simulations=30,
    mcts_rollout_steps=100,
    belief_model=None,
    temperature=1.0,
    risk_aversion=0.0,
    mcts_player=1,
):
    if seed is not None:
        np.random.seed(seed)
    samples_x = []
    samples_y = []
    samples_mask = []

    for _ in range(num_games):
        game = big2Game.big2Game()
        # bootstrap history vector shape
        history = [_action_vector(1, None, 0, False)]
        turn = 0
        while not game.gameOver and turn < max_turns:
            x, y, mask = _build_sample(game, history, history_len)
            samples_x.append(x)
            samples_y.append(y)
            samples_mask.append(mask)

            if policy == "random":
                action = _choose_random_action(game)
            elif policy == "mcts":
                if game.playersGo == mcts_player:
                    action = mcts_search(
                        game,
                        simulations=mcts_simulations,
                        max_rollout_steps=mcts_rollout_steps,
                        belief_model=belief_model,
                        temperature=temperature,
                        risk_aversion=risk_aversion,
                    )
                else:
                    action = _choose_heuristic_action(game)
            else:
                action = _choose_heuristic_action(game)
            player = game.playersGo
            if action == -1:
                hand = None
                game.updateGame(-1)
            else:
                try:
                    opt, nC, hand = _decode_action(game, action)
                except IndexError:
                    action = _choose_heuristic_action(game)
                    if action == -1:
                        hand = None
                        game.updateGame(-1)
                        history.append(_action_vector(player, hand, game.passCount, game.control))
                        turn += 1
                        continue
                    opt, nC, hand = _decode_action(game, action)
                game.updateGame(opt, nC)
            history.append(_action_vector(player, hand, game.passCount, game.control))
            turn += 1
    return (
        np.stack(samples_x, axis=0),
        np.stack(samples_y, axis=0),
        np.stack(samples_mask, axis=0),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--history-len", type=int, default=20)
    parser.add_argument("--max-turns", type=int, default=300)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", type=str, default="ml/belief_dataset.npz")
    parser.add_argument(
        "--policy", type=str, choices=["heuristic", "random", "mcts"], default="heuristic"
    )
    parser.add_argument("--mcts-simulations", type=int, default=30)
    parser.add_argument("--mcts-rollout-steps", type=int, default=100)
    parser.add_argument("--belief-model", type=str, default="")
    parser.add_argument("--belief-data", type=str, default="ml/belief_dataset.npz")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--risk-aversion", type=float, default=0.0)
    parser.add_argument("--mcts-player", type=int, default=1)
    args = parser.parse_args()

    belief_model = None
    if args.belief_model:
        data = np.load(args.belief_data)
        belief_model = BeliefNet(data["x"].shape[1])
        payload = torch.load(args.belief_model, map_location="cpu")
        belief_model.load_state_dict(payload["model_state"])
        belief_model.eval()

    x, y, mask = generate_dataset(
        args.games,
        args.history_len,
        args.max_turns,
        args.seed,
        args.policy,
        mcts_simulations=args.mcts_simulations,
        mcts_rollout_steps=args.mcts_rollout_steps,
        belief_model=belief_model,
        temperature=args.temperature,
        risk_aversion=args.risk_aversion,
        mcts_player=args.mcts_player,
    )
    np.savez(args.out, x=x, y=y, mask=mask, history_len=args.history_len)
    print(f"saved: {args.out}  samples={x.shape[0]}")


if __name__ == "__main__":
    main()
