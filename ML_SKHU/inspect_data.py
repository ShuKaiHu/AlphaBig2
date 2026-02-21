import argparse
import numpy as np
import torch

from ML_SKHU.belief import BeliefModel
from ML_SKHU.policy_value import PolicyValueModel
from ML_SKHU.features import belief_input_dim, encode_belief_input, encode_policy_input, belief_targets
from ML_SKHU.selfplay import run_selfplay_episode
import enumerateOptions


def _card_label(card_id):
    value = int(np.ceil(card_id / 4.0))
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
    suit_idx = (card_id - 1) % 4
    suit = ["C", "D", "H", "S"][suit_idx]
    return f"{rank}{suit}"


def _cards_from_mask(mask):
    return [_card_label(i + 1) for i, v in enumerate(mask) if v > 0.5]

def _action_to_cards(action):
    if action == enumerateOptions.passInd:
        return "pass"
    opt, n_cards = enumerateOptions.getOptionNC(action)
    if n_cards == 1:
        return [_card_label(opt + 1)]
    if n_cards == 2:
        return [_card_label(i + 1) for i in enumerateOptions.inverseTwoCardIndices[opt]]
    if n_cards == 3:
        return [_card_label(i + 1) for i in enumerateOptions.inverseThreeCardIndices[opt]]
    if n_cards == 4:
        return [_card_label(i + 1) for i in enumerateOptions.inverseFourCardIndices[opt]]
    if n_cards == 5:
        return [_card_label(i + 1) for i in enumerateOptions.inverseFiveCardIndices[opt]]
    return "unknown"


def _slice_features(vec):
    # Follow features.py ordering
    idx = 0
    my_hand = vec[idx:idx + 52]; idx += 52
    played = vec[idx:idx + 52]; idx += 52
    remaining = vec[idx:idx + 4]; idx += 4
    current_player = vec[idx:idx + 4]; idx += 4
    control = vec[idx:idx + 1]; idx += 1
    must_play_club3 = vec[idx:idx + 1]; idx += 1
    passed = vec[idx:idx + 4]; idx += 4
    last_player = vec[idx:idx + 4]; idx += 4
    prev_type = vec[idx:idx + 7]; idx += 7
    prev_rank = vec[idx:idx + 13]; idx += 13
    prev_suit = vec[idx:idx + 4]; idx += 4
    prev_straight = vec[idx:idx + 10]; idx += 10
    return {
        "my_hand": my_hand,
        "played": played,
        "remaining": remaining,
        "current_player": current_player,
        "control": control,
        "must_play_club3": must_play_club3,
        "passed": passed,
        "last_player": last_player,
        "prev_type": prev_type,
        "prev_rank": prev_rank,
        "prev_suit": prev_suit,
        "prev_straight": prev_straight,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--belief-ckpt", type=str, default="")
    parser.add_argument("--policy-ckpt", type=str, default="")
    parser.add_argument("--simulations", type=int, default=25)
    args = parser.parse_args()

    b_input_dim = belief_input_dim()
    p_input_dim = b_input_dim + 52 * 3

    belief_model = BeliefModel(b_input_dim).to(args.device)
    policy_value_model = PolicyValueModel(p_input_dim).to(args.device)

    if args.belief_ckpt:
        belief_model.load_state_dict(torch.load(args.belief_ckpt, map_location=args.device))
    if args.policy_ckpt:
        policy_value_model.load_state_dict(torch.load(args.policy_ckpt, map_location=args.device))

    belief_model.eval()
    policy_value_model.eval()

    belief_data, policy_data = run_selfplay_episode(
        belief_model=belief_model,
        policy_value_model=policy_value_model,
        n_simulations=args.simulations,
        device=args.device,
    )

    belief_sample = belief_data[0]
    belief_in, b_target, b_mask = belief_sample[:3]
    policy_in, policy_target, value_target = policy_data[0]

    parts = _slice_features(belief_in)
    print("Belief input shape:", belief_in.shape)
    print("My hand:", _cards_from_mask(parts["my_hand"]))
    print("Played cards:", _cards_from_mask(parts["played"]))
    print("Remaining (P1..P4):", np.round(parts["remaining"] * 13).astype(int).tolist())
    print("Current player:", int(np.argmax(parts["current_player"])) + 1)
    print("Control:", bool(parts["control"][0]))
    print("Must play 3C:", bool(parts["must_play_club3"][0]))
    print("Passed this round:", (parts["passed"] > 0.5).astype(int).tolist())
    print("Last player:", int(np.argmax(parts["last_player"])) + 1)
    print("Prev hand type:", np.argmax(parts["prev_type"]))
    print("Prev hand rank:", int(np.argmax(parts["prev_rank"])) + 1)
    print("Prev hand suit:", int(np.argmax(parts["prev_suit"])))
    print("Prev straight idx:", int(np.argmax(parts["prev_straight"])))
    valid_targets = b_target[(b_mask > 0) & (b_target >= 0)]
    print("Belief target counts (P2,P3,P4):", np.bincount(valid_targets, minlength=3).tolist())
    print("Belief unknown cards:", int(b_mask.sum()))

    print("\nPolicy input shape:", policy_in.shape)
    top5 = np.argsort(-policy_target)[:5]
    print("Policy target top5 actions:", top5.tolist())
    print("Policy target top5 probs:", np.round(np.sort(policy_target)[-5:][::-1], 4).tolist())
    print("Policy target top5 cards:", [_action_to_cards(int(a)) for a in top5])
    print("Value target:", float(value_target))


if __name__ == "__main__":
    main()
