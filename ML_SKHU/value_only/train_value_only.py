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
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save", required=True)
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--eval-games", type=int, default=0)
    parser.add_argument("--eval-actor", choices=["heuristic", "random"], default="heuristic")
    parser.add_argument("--freeze-policy", action="store_true")
    args = parser.parse_args()

    data = np.load(args.data)
    obs = torch.tensor(data["obs"], dtype=torch.float32).to(args.device)
    masks = torch.tensor(data["mask"], dtype=torch.float32).to(args.device)
    values = torch.tensor(data["values"], dtype=torch.float32).to(args.device)

    input_dim = obs.shape[1]
    model = PolicyValueModel(input_dim, hidden_dim=256).to(args.device)

    if args.freeze_policy:
        for p in model.policy_head.parameters():
            p.requires_grad = False

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    for epoch in range(args.epochs):
        idx = torch.randperm(obs.size(0))
        total_loss = 0.0
        for i in range(0, obs.size(0), args.batch_size):
            batch_idx = idx[i : i + args.batch_size]
            batch_x = obs[batch_idx]
            batch_mask = masks[batch_idx]
            batch_values = values[batch_idx]

            _, pred = model(batch_x, batch_mask)
            loss = nn.functional.mse_loss(pred.view(-1), batch_values.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_x.size(0)

        avg_loss = total_loss / obs.size(0)
        log_this = epoch % 10 == 0 or epoch == args.epochs - 1
        eval_this = (
            args.eval_every > 0
            and args.eval_games > 0
            and (epoch % args.eval_every == 0 or epoch == args.epochs - 1)
        )
        if log_this or eval_this:
            line = f"epoch={epoch} value_loss={avg_loss:.4f}"
            if eval_this:
                from ML_SKHU.value_only.eval_value_only import eval_model

                mse, mae, rmse, corr, _, _ = eval_model(
                    model,
                    games=args.eval_games,
                    actor=args.eval_actor,
                    device=args.device,
                )
                line += f" eval_mse={mse:.4f} eval_mae={mae:.4f} eval_corr={corr:.4f}"
            print(line)

    torch.save(model.state_dict(), args.save)
    print(f"saved {args.save}")


if __name__ == "__main__":
    main()
