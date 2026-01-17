import argparse
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


class BeliefNet(nn.Module):
    def __init__(self, input_dim, hidden_sizes=(512, 256)):
        super().__init__()
        h1, h2 = hidden_sizes
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1),
            GELU(),
            nn.Linear(h1, h2),
            GELU(),
            nn.Linear(h2, 52 * 3),
        )

    def forward(self, x):
        logits = self.net(x)
        return logits.view(-1, 52, 3)


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


def masked_cross_entropy(logits, targets, mask):
    # logits: [B, 52, 3], targets: [B, 52], mask: [B, 52]
    b, n, _ = logits.shape
    logits = logits.view(b * n, 3)
    targets = targets.view(b * n)
    mask = mask.view(b * n)
    valid = mask > 0.5
    if valid.sum() == 0:
        return logits.mean() * 0.0
    return nn.functional.cross_entropy(logits[valid], targets[valid])


def train(args):
    data = np.load(args.data)
    x = torch.tensor(data["x"], dtype=torch.float32)
    y = torch.tensor(data["y"], dtype=torch.long)
    mask = torch.tensor(data["mask"], dtype=torch.float32)

    n = x.shape[0]
    split = int(n * args.train_split)
    train_set = TensorDataset(x[:split], y[:split], mask[:split])
    val_set = TensorDataset(x[split:], y[split:], mask[split:])

    dataset = train_set
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = BeliefNet(x.shape[1]).to("cpu")
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_acc = -1.0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        for bx, by, bm in loader:
            logits = model(bx)
            loss = masked_cross_entropy(logits, by, bm)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.item() * bx.size(0)
        avg_loss = total_loss / max(len(dataset), 1)
        val_acc = evaluate(model, val_set, args.batch_size) if len(val_set) > 0 else 0.0
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        if epoch % args.log_every == 0:
            print(f"epoch={epoch} loss={avg_loss:.4f} val_acc={val_acc:.3f}")

    if args.save and best_state is not None:
        torch.save({"model_state": best_state, "best_acc": best_acc}, args.save)
        print(f"saved best: {args.save} acc={best_acc:.3f}")

    if args.save_last:
        torch.save({"model_state": model.state_dict()}, args.save_last)
        print(f"saved last: {args.save_last}")


def evaluate(model, dataset, batch_size):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    correct = 0
    total = 0
    with torch.no_grad():
        for bx, by, bm in loader:
            logits = model(bx)
            preds = torch.argmax(logits, dim=-1)
            valid = bm > 0.5
            correct += (preds[valid] == by[valid]).sum().item()
            total += valid.sum().item()
    if total == 0:
        return 0.0
    return correct / total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="ml/belief_dataset.npz")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--train-split", type=float, default=0.9)
    parser.add_argument("--save", type=str, default="ml/belief_model_best.pt")
    parser.add_argument("--save-last", type=str, default="")
    train(parser.parse_args())
