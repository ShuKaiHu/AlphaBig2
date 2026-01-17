import argparse
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ml.policy import PolicyNet


def masked_kl_loss(logits, target_probs, mask):
    # logits: [B, 1695], target_probs: [B, 1695], mask: [B, 1695]
    mask_bool = mask > 0.5
    logits = logits.masked_fill(~mask_bool, -1.0e9)
    log_probs = nn.functional.log_softmax(logits, dim=-1)
    target = target_probs * mask
    target = target / (target.sum(dim=-1, keepdim=True) + 1e-8)
    return nn.functional.kl_div(log_probs, target, reduction="batchmean")


def train(args):
    data = np.load(args.data)
    obs = torch.tensor(data["obs"], dtype=torch.float32)
    mask = torch.tensor(data["mask"], dtype=torch.float32)
    action_probs = torch.tensor(data["action_probs"], dtype=torch.float32)

    n = obs.shape[0]
    split = int(n * args.train_split)
    train_set = TensorDataset(obs[:split], mask[:split], action_probs[:split])
    val_set = TensorDataset(obs[split:], mask[split:], action_probs[split:])
    loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

    model = PolicyNet().to("cpu")
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_acc = -1.0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        for bx, bm, bp in loader:
            logits = model(bx, bm)
            loss = masked_kl_loss(logits, bp, bm)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.item() * bx.size(0)
        avg_loss = total_loss / max(len(train_set), 1)
        val_acc = evaluate(model, val_set, args.batch_size)
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        if epoch % args.log_every == 0:
            print(f"epoch={epoch} loss={avg_loss:.4f} val_acc={val_acc:.3f}")

    if args.save and best_state is not None:
        torch.save({"model_state": best_state, "best_acc": best_acc}, args.save)
        print(f"saved best: {args.save} acc={best_acc:.3f}")


def evaluate(model, dataset, batch_size):
    if len(dataset) == 0:
        return 0.0
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    correct = 0
    total = 0
    with torch.no_grad():
        for bx, bm, bp in loader:
            logits = model(bx, bm)
            preds = torch.argmax(logits, dim=-1)
            target = torch.argmax(bp, dim=-1)
            correct += (preds == target).sum().item()
            total += target.size(0)
    return correct / max(total, 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="ml/imitation_dataset.npz")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--train-split", type=float, default=0.9)
    parser.add_argument("--save", type=str, default="ml/imitation_policy.pt")
    train(parser.parse_args())
