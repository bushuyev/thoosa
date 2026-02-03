import torch
import torch.nn as nn
import torch.nn.functional as F

def lengths_to_mask(lengths, T):
    # lengths: [B]
    t = torch.arange(T, device=lengths.device)[None, :]
    return (t < lengths[:, None])  # [B, T] bool


class AttnPool1d(nn.Module):
    """
    h:    [B, C, T]
    mask: [B, T] bool (True = valid)
    returns:
      pooled: [B, C]
      attn:   [B, T]
    """
    def __init__(self, C, attn_hidden=128):
        super().__init__()
        self.proj = nn.Conv1d(C, attn_hidden, kernel_size=1)
        self.v    = nn.Conv1d(attn_hidden, 1, kernel_size=1, bias=False)

    def forward(self, h, mask=None):
        p = self.proj(h)
        t = torch.tanh(p)
        scores = self.v(t)  # [B, T]
        print(f"p.shape={p.shape} t.shape={t.shape} scores={scores.shape}")

        scores = scores.squeeze(1);
        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9)
        attn = torch.softmax(scores, dim=-1)                  # [B, T]
        attn_unsqueeze = attn.unsqueeze(1)
        print(f"scores.shape={scores.shape} attn={attn.shape} h.shape={h.shape} attn_unsqueeze.shape={attn_unsqueeze.shape}")
        pooled = (h * attn_unsqueeze).sum(dim=-1)

        print(f"pooled={pooled.shape}")# [B, C]
        return pooled, attn


class MotifCNNWithAttn(nn.Module):
    """
    Example parent module that:
      - extracts features with a Conv1d stack
      - pools over time with AttnPool1d
      - outputs one logit per sequence
    """
    def __init__(self, n_motifs, C=64, attn_hidden=128):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Conv1d(n_motifs, C, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(C, C, kernel_size=5, padding=2),
            nn.ReLU(),
        )
        self.pool = AttnPool1d(C, attn_hidden=attn_hidden)
        self.head = nn.Linear(C, 1)

    def forward(self, x, lengths, return_attn=False):
        # x: [B, M, T_max]
        h = self.feat(x)                                # [B, C, T_max]
        mask = lengths_to_mask(lengths, h.shape[-1])    # [B, T_max]
        pooled, attn = self.pool(h, mask)               # [B, C], [B, T_max]
        logits = self.head(pooled).squeeze(-1)          # [B]
        if return_attn:
            return logits, attn
        return logits


# ----------- tiny runnable demo -----------
if __name__ == "__main__":
    B, M = 3, 5
    lengths = torch.tensor([8, 5, 10])     # variable lengths in batch
    T_max = int(lengths.max())

    x = torch.zeros(B, M, T_max)
    for i, L in enumerate(lengths.tolist()):
        x[i, :, :L] = torch.rand(M, L)     # pretend these are motif match probabilities

    model = MotifCNNWithAttn(n_motifs=M, C=32, attn_hidden=64)
    logits, attn = model(x, lengths, return_attn=True)

    print("logits:", logits)               # [B]
    print("attn shape:", attn.shape)       # [B, T_max]
    print("attn sums:", attn.sum(dim=-1))  # should be ~1.0 for each sample
