import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# -----------------------------
# 1) Deterministic stream builder
# -----------------------------
def make_x_1(
        length: int,
        hit_idxs: list[int],
        hit_vals: list[float] | None = None,
        baseline: float = 0.0,
        clamp_01: bool = True,
) -> torch.Tensor:
    """
    Create a length-L 1D tensor with spikes at exact indices.
    - hit_idxs: e.g. [1,3,5]
    - hit_vals: optional amplitudes per index (same length as hit_idxs)
    """
    x = torch.full((length,), baseline, dtype=torch.float32)

    if hit_vals is None:
        hit_vals = [0.95] * len(hit_idxs)

    assert len(hit_vals) == len(hit_idxs)

    for i, v in zip(hit_idxs, hit_vals):
        if 0 <= i < length:
            x[i] = float(v)

    if clamp_01:
        x = x.clamp(0.0, 1.0)
    return x


def shift_indices(idxs: list[int], shift: int, L: int) -> list[int]:
    """Deterministically shift indices by +shift (dropping those that would go out of range)."""
    out = []
    for i in idxs:
        j = i + shift
        if 0 <= j < L:
            out.append(j)
    return out


# -----------------------------
# 2) Deterministic dataset from templates
# -----------------------------
def make_sample_from_template(
        C: int,
        L: int,
        per_channel_hit_idxs: list[list[int]],
        shift: int = 0,
        amp_scale: float = 1.0,
) -> torch.Tensor:
    """
    Build x: [C,L] from a template of indices for each channel.
    """
    assert len(per_channel_hit_idxs) == C
    x = torch.zeros(C, L, dtype=torch.float32)

    # deterministic per-hit amplitude schedule (optional):
    # e.g. [0.9, 0.95, 1.0, 0.92, ...] repeating
    base_vals = [0.90, 0.95, 1.00, 0.92, 0.98]

    for c in range(C):
        idxs = shift_indices(per_channel_hit_idxs[c], shift, L)
        vals = [(base_vals[k % len(base_vals)] * amp_scale) for k in range(len(idxs))]
        x[c] = make_x_1(L, idxs, vals)

    return x


def make_dataset(
        n_samples: int,
        C: int,
        L: int,
        tpl0: list[list[int]],
        tpl1: list[list[int]],
        # deterministic shift schedule: keeps order, moves positions
        shift_cycle: list[int] = [0, 1, 2, 1],   # repeat this per sample
        # deterministic amplitude schedule (optional)
        amp_cycle: list[float] = [1.0, 0.95, 1.05, 1.0],
) -> TensorDataset:
    x = torch.empty(n_samples, C, L, dtype=torch.float32)
    y = torch.empty(n_samples, 1, dtype=torch.float32)

    for i in range(n_samples):
        label = i % 2  # strictly alternating 0,1,0,1,... (deterministic)
        shift = shift_cycle[i % len(shift_cycle)]
        amp = amp_cycle[i % len(amp_cycle)]

        tpl = tpl1 if label == 1 else tpl0
        x[i] = make_sample_from_template(C, L, tpl, shift=shift, amp_scale=amp)
        y[i, 0] = float(label)

    return TensorDataset(x, y)


# -----------------------------
# 3) Attention pooling model over time
# -----------------------------
class AttnPool1d(nn.Module):
    """
    Input:  x [B,C,L]
    Output: logits [B,1], attn [B,L]
    """
    def __init__(self, C: int, hidden: int = 32):
        super().__init__()
        self.feat = nn.Conv1d(C, hidden, kernel_size=1)      # [B,C,L] -> [B,H,L]
        self.score = nn.Conv1d(hidden, 1, kernel_size=1)     # [B,H,L] -> [B,1,L]
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor):
        h = torch.tanh(self.feat(x))                         # [B,H,L]
        s = self.score(h).squeeze(1)                         # [B,L]
        a = torch.softmax(s, dim=-1)                         # [B,L]
        ctx = (h * a.unsqueeze(1)).sum(dim=-1)               # [B,H]
        logits = self.fc(ctx)                                # [B,1]
        return logits, a


# -----------------------------
# 4) Train + inspect attention peaks
# -----------------------------
@torch.no_grad()
def inspect_attention(attn: torch.Tensor, topk: int = 6):
    # attn: [B,L]
    vals, idxs = torch.topk(attn, k=topk, dim=-1)
    return idxs.cpu(), vals.cpu()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    C, L = 4, 32

    # Class templates: per channel hit indices (deterministic, fixed order)
    # Think: each channel = one motif detector stream.
    tpl_class0 = [
        [2, 4, 6],       # channel 0
        [10, 11],        # channel 1
        [],              # channel 2
        [20],            # channel 3
    ]
    tpl_class1 = [
        [1, 3, 5],       # channel 0
        [12, 14],        # channel 1
        [7],             # channel 2
        [],              # channel 3
    ]

    ds = make_dataset(
        n_samples=2000,
        C=C,
        L=L,
        tpl0=tpl_class0,
        tpl1=tpl_class1,
        shift_cycle=[0, 1, 2, 1],        # deterministic movement over samples
        amp_cycle=[1.0, 0.95, 1.05, 1.0] # deterministic amp variation
    )

    dl = DataLoader(ds, batch_size=64, shuffle=False)  # keep deterministic order

    model = AttnPool1d(C=C, hidden=32).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    # train
    for epoch in range(1, 11):
        model.train()
        total = 0.0
        nb = 0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            logits, attn = model(xb)
            loss = loss_fn(logits, yb)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total += loss.item()
            nb += 1

        if epoch == 1 or epoch % 2 == 0:
            print(f"epoch {epoch:02d} | loss {total / nb:.4f}")

    # inspect attention on first batch
    model.eval()
    xb, yb = next(iter(dl))
    xb = xb.to(device)
    logits, attn = model(xb)
    probs = torch.sigmoid(logits).squeeze(1).cpu()
    top_idxs, top_vals = inspect_attention(attn.cpu(), topk=6)

    print("\nFirst 8 samples:")
    for i in range(8):
        print(
            f"y={int(yb[i].item())}  p={probs[i].item():.3f}  "
            f"attn_top_idx={top_idxs[i].tolist()}  attn_top_val={[round(v,3) for v in top_vals[i].tolist()]}"
        )


if __name__ == "__main__":
    main()
