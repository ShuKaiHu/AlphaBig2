import argparse
import json

import numpy as np
import torch

import big2Game

from ML_AB.actions import action_to_string
from ML_AB.agents import ModelAgent
from ML_AB.eval import load_model
from ML_AB.state import action_mask
from ML_AB.utils import device_from_arg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()
    device = device_from_arg(args.device)
    model = load_model(args.ckpt, device)
    agent = ModelAgent(model, device=device, temperature=0.0)

    # Demo recommendation from a freshly dealt local game. The online adapter
    # should construct an equivalent big2Game state and call ModelAgent.
    game = big2Game.big2Game()
    logits, value = agent.action_logits(game, game.playersGo)
    valid = np.flatnonzero(action_mask(game) > 0)
    top = valid[np.argsort(-logits[valid])][: args.topk]
    print(
        json.dumps(
            {
                "player": int(game.playersGo),
                "value": value,
                "recommendations": [
                    {"action": int(a), "cards": action_to_string(int(a)), "logit": float(logits[int(a)])}
                    for a in top
                ],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
