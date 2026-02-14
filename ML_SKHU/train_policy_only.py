import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ML_SKHU.policy_value import PolicyValueModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save", required=True)
    parser.add_argument("--value-weight", type=float, default=1.0)
    parser.add_argument("--reinforce", action="store_true")
    args = parser.parse_args()

    data = np.load(args.data)
    obs = torch.tensor(data["obs"], dtype=torch.float32).to(args.device)
    masks = torch.tensor(data["mask"], dtype=torch.float32).to(args.device)
    actions = torch.tensor(data["actions"], dtype=torch.long).to(args.device)
    rewards = torch.tensor(data["values"], dtype=torch.float32).to(args.device)

    p_input_dim = obs.shape[1]
    model = PolicyValueModel(p_input_dim, hidden_dim=256).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        idx = torch.randperm(obs.size(0))
        total_loss = 0.0
        for i in range(0, obs.size(0), args.batch_size):
            batch_idx = idx[i : i + args.batch_size]
            batch_x = obs[batch_idx]
            batch_mask = masks[batch_idx]
            batch_actions = actions[batch_idx]
            batch_rewards = rewards[batch_idx]

            logits, values = model(batch_x, batch_mask)
            log_probs = nn.functional.log_softmax(logits, dim=-1)
            selected_log_probs = log_probs[torch.arange(log_probs.size(0)), batch_actions]
            action_loss = -selected_log_probs.mean()
            value_loss = nn.functional.mse_loss(values, batch_rewards)
            loss = action_loss + args.value_weight * value_loss
            if args.reinforce:
                with torch.no_grad():
                    baseline = values.detach().squeeze(-1)
                advantage = (batch_rewards - baseline).detach()
                reinforce_loss = -(advantage * selected_log_probs).mean()
                loss = loss + reinforce_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_x.size(0)

        avg_loss = total_loss / obs.size(0)
        if epoch % 5 == 0:
            print(f"epoch={epoch} loss={avg_loss:.4f}")

    torch.save(model.state_dict(), args.save)
    print(f"saved {args.save}")


if __name__ == "__main__":
    main()
