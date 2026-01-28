#!/usr/bin/env python3
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


def collate_pad(xs, ys):
    lengths = torch.tensor([x.shape[1] for x in xs], dtype=torch.long)
    T_max = int(lengths.max())
    M = xs[0].shape[0]

    x_pad = torch.zeros(len(xs), M, T_max, dtype=xs[0].dtype)
    for i, x in enumerate(xs):
        x_pad[i, :, : x.shape[1]] = x

    y = torch.tensor(ys, dtype=torch.float32)
    return x_pad, lengths, y


def make_mask(lengths: torch.Tensor, T: int) -> torch.Tensor:
    t = torch.arange(T, device=lengths.device)[None, :]
    return (t < lengths[:, None])[:, None, :]


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    x = x.masked_fill(~mask, 0.0)
    denom = mask.sum(dim=-1).clamp_min(1)
    return x.sum(dim=-1) / denom


def masked_max(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    x = x.masked_fill(~mask, float("-inf"))
    return x.max(dim=-1).values


class MotifCNN(nn.Module):
    def __init__(self, n_motifs: int, hidden: int = 64, dropout: float = 0.2):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Conv1d(n_motifs, hidden, 5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, 5, padding=2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, hidden * 2, 3, padding=1),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(4 * hidden, 128),  # max+mean over time -> 2*(2H)=4H
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:

        h = self.feat(x)  # [B, 2H, T_max]
        print(f"x={x.size()} h={h.size()}")

        mask = make_mask(lengths, h.shape[-1])
        pooled = torch.cat([masked_max(h, mask), masked_mean(h, mask)], dim=1)  # [B,4H]
        return self.head(pooled).squeeze(-1)  # [B]


def generate_batch(B: int, M: int, rng: random.Random):
    # fixed length choices (demo only). In real life: you already *have* xs with varying T.
    # T_choices = [10_000, 25_000, 60_000, 180_000]  # include 600_000 if you dare 🙂
    T_choices = [100, 250, 600, 1800]  # include 600_000 if you dare 🙂
    xs, ys = [], []

    for _ in range(B):
        T = rng.choice(T_choices)
        y = 1 if rng.random() < 0.5 else 0
        ys.append(y)

        x = torch.rand(M, T) * 0.3  # mostly low probabilities
        if y == 1:
            center = rng.randrange(T)
            width = max(10, T // 80)
            t = torch.arange(T)
            bump = torch.exp(-0.5 * ((t - center) / width) ** 2)
            x[0] = (x[0] + 0.8 * bump).clamp(0.0, 1.0)

        xs.append(x)

    return xs, ys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--n_motifs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    model = MotifCNN(args.n_motifs).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    crit = nn.BCEWithLogitsLoss()

    for step in range(1, args.steps + 1):
        xs, ys = generate_batch(args.batch_size, args.n_motifs, rng)
        x_pad, lengths, y = collate_pad(xs, ys)

        x_pad = x_pad.to(device)
        lengths = lengths.to(device)
        y = y.to(device)

        opt.zero_grad(set_to_none=True)
        logits = model(x_pad, lengths)
        loss = crit(logits, y)
        loss.backward()
        opt.step()

        if step % 5 == 0 or step == 1:
            with torch.no_grad():
                probs = torch.sigmoid(logits)
                acc = ((probs >= 0.5).float() == y).float().mean().item()
            print(f"step {step:03d} loss={loss.item():.4f} acc={acc:.3f} "
                  f"T_max={x_pad.shape[-1]} lengths={lengths.tolist()}")

    # inference example
    xs, ys = generate_batch(args.batch_size, args.n_motifs, rng)
    x_pad, lengths, _ = collate_pad(xs, ys)
    with torch.no_grad():
        probs = torch.sigmoid(model(x_pad.to(device), lengths.to(device))).cpu()
    print("\nInference demo:")
    print("y_true:", ys)
    print("p_hat :", [round(p, 4) for p in probs.tolist()])


if __name__ == "__main__":
    main()
