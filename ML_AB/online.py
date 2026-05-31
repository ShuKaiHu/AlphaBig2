import argparse
import json
from types import SimpleNamespace

import numpy as np

import big2Game

from ML_AB.actions import action_to_string
from ML_AB.agents import ModelAgent, RerankAgent
from ML_AB.eval import load_model
from ML_AB.state import action_mask
from ML_AB.utils import device_from_arg


def _cards_array(cards):
    return np.array([int(c) for c in cards], dtype=int)


def build_public_game(
    my_hand,
    perspective_player=1,
    current_player=1,
    opponent_counts=None,
    played_cards=None,
    played_cards_by_player=None,
    last_hand=None,
    last_player=None,
    control=True,
    passed=None,
    action_history=None,
):
    """Build the minimal game-like object needed by the model from public data.

    This is the bridge for an online client. It does not need hidden opponent
    cards; opponent hands are dummy arrays with the right lengths.
    """
    if int(current_player) != int(perspective_player):
        raise ValueError("recommendation is only valid when it is the model player's turn")
    opponent_counts = opponent_counts or {}
    passed = passed or {}
    played_cards = played_cards or []
    played_cards_by_player = played_cards_by_player or {}

    game = big2Game.big2Game()
    game.currentHands = {}
    for p in range(1, 5):
        if p == int(perspective_player):
            game.currentHands[p] = np.sort(_cards_array(my_hand))
        else:
            game.currentHands[p] = np.zeros((int(opponent_counts.get(str(p), opponent_counts.get(p, 0))),), dtype=int)

    game.cardsPlayed = np.zeros((4, 52), dtype=int)
    for cid in played_cards:
        game.cardsPlayed[0, int(cid) - 1] = 1
    for p, cards in played_cards_by_player.items():
        for cid in cards:
            game.cardsPlayed[int(p) - 1, int(cid) - 1] = 1

    game.playersGo = int(current_player)
    game.control = 1 if control else 0
    game.passCount = int(sum(1 for p in range(1, 5) if bool(passed.get(str(p), passed.get(p, False)))))
    game.passedThisRound = {
        p: bool(passed.get(str(p), passed.get(p, False))) for p in range(1, 5)
    }
    game.lastPlayedPlayer = int(last_player or perspective_player)
    game.handsPlayed = {}
    game.goIndex = 1
    if last_hand:
        game.handsPlayed[1] = big2Game.handPlayed(_cards_array(last_hand), game.lastPlayedPlayer)
        game.goIndex = 2
    game.mustPlayClub3 = (len(played_cards) == 0 and not played_cards_by_player and 1 in set(int(c) for c in my_hand))
    game.club3Player = int(perspective_player)
    game.actionHistory = action_history or []
    game.gameOver = 0
    return game


def recommend_from_observation(
    model,
    observation,
    device="cpu",
    topk=5,
    agent_type="model",
    rerank_kwargs=None,
):
    game = build_public_game(**observation)
    if agent_type == "rerank":
        agent = RerankAgent(model, device=device, temperature=0.0, **(rerank_kwargs or {}))
    else:
        agent = ModelAgent(model, device=device, temperature=0.0)
    logits, value = agent.action_logits(game, game.playersGo)
    valid = np.flatnonzero(action_mask(game) > 0)
    top = valid[np.argsort(-logits[valid])][:topk]
    return {
        "player": int(game.playersGo),
        "agent": agent_type,
        "value": float(value),
        "recommendations": [
            {"action": int(a), "cards": action_to_string(int(a)), "logit": float(logits[int(a)])}
            for a in top
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--observation-json", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--agent", choices=["model", "rerank"], default="model")
    parser.add_argument("--control-five-bonus", type=float, default=1.2)
    parser.add_argument("--card-count-bonus", type=float, default=0.12)
    parser.add_argument("--finish-bonus", type=float, default=3.0)
    parser.add_argument("--urgent-opponent-count", type=int, default=3)
    parser.add_argument("--urgent-five-bonus", type=float, default=1.0)
    parser.add_argument("--preserve-five-card-penalty", type=float, default=0.25)
    args = parser.parse_args()
    device = device_from_arg(args.device)
    model = load_model(args.ckpt, device)
    with open(args.observation_json, "r") as f:
        observation = json.load(f)
    print(
        json.dumps(
            recommend_from_observation(
                model,
                observation,
                device=device,
                topk=args.topk,
                agent_type=args.agent,
                rerank_kwargs={
                    "control_five_bonus": args.control_five_bonus,
                    "card_count_bonus": args.card_count_bonus,
                    "finish_bonus": args.finish_bonus,
                    "urgent_opponent_count": args.urgent_opponent_count,
                    "urgent_five_bonus": args.urgent_five_bonus,
                    "preserve_five_card_penalty": args.preserve_five_card_penalty,
                },
            ),
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
