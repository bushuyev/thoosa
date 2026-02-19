import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, bias: bool = True):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.proj = nn.Linear(d_model, d_model, bias=bias)

        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        self.register_buffer("causal_mask", mask, persistent=False)

        # Debug caches (filled on forward)
        self.last_in = None            # [B,T,d_model]  (input to attention = LN(residual))
        self.last_attn = None          # [B,H,T,T]
        self.last_v = None             # [B,H,T,d_head]
        self.last_head_out = None      # [B,H,T,d_head]  (attn @ v)
        self.last_out_before_proj = None  # [B,T,d_model] (heads concatenated)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        self.last_in = x.detach()

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # [B,H,T,dh]
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)  # [B,H,T,T]
        mask = self.causal_mask[:T, :T]
        scores = scores.masked_fill(~mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)  # [B,H,T,T]
        head_out = attn @ v               # [B,H,T,dh]

        out = head_out.transpose(1, 2).contiguous().view(B, T, C)  # concat heads -> [B,T,d_model]

        # Cache debug tensors
        self.last_attn = attn.detach()
        self.last_v = v.detach()
        self.last_head_out = head_out.detach()
        self.last_out_before_proj = out.detach()

        return self.proj(out)


class MLP(nn.Module):
    def __init__(self, d_model: int, d_mlp: int, bias: bool = True):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_mlp, bias=bias)
        self.fc2 = nn.Linear(d_mlp, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_mlp: int, max_seq_len: int, bias: bool = True):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, max_seq_len, bias=bias)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, d_mlp, bias=bias)

        # Debug caches around the residual stream
        self.last_resid_pre_attn = None     # x before adding attn
        self.last_resid_post_attn = None    # x after adding attn (before mlp)
        self.last_attn_delta = None         # attn(LN(x)) output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.last_resid_pre_attn = x.detach()
        delta = self.attn(self.ln1(x))   # <-- "read" from residual stream (LN(x))
        self.last_attn_delta = delta.detach()
        x = x + delta                    # <-- "write" into residual stream (add to x)
        print(f"delta={delta.size()}")
        self.last_resid_post_attn = x.detach()

        x = x + self.mlp(self.ln2(x))
        return x


class TinyDecoderOnlyTransformerLM(nn.Module):
    def __init__(self, d_vocab: int, max_seq_len: int, n_layers: int, d_model: int, n_heads: int, d_mlp: int):
        super().__init__()
        self.d_vocab = d_vocab
        self.max_seq_len = max_seq_len

        self.tok_embed = nn.Embedding(d_vocab, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, d_model))

        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_mlp, max_seq_len) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.unembed = nn.Linear(d_model, d_vocab)

        nn.init.normal_(self.pos_embed, std=0.02)

    def forward(self, idx: torch.Tensor):
        B, T = idx.shape
        assert T <= self.max_seq_len

        x = self.tok_embed(idx) + self.pos_embed[:, :T, :]
        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.unembed(x)
        return logits


def inspect_first_block_attention(model: TinyDecoderOnlyTransformerLM, idx: torch.Tensor, dest_t: int):
    model.eval()
    with torch.no_grad():
        _ = model(idx)

    blk = model.blocks[0]
    attn = blk.attn

    # Pull caches (assume batch=1, head=0 for simplicity)
    A = attn.last_attn[0, 0]          # [T,T]
    V = attn.last_v[0, 0]             # [T,dh]
    head_out = attn.last_head_out[0, 0]  # [T,dh]

    # Show where dest_t is reading from
    weights = A[dest_t]               # [T]
    topw, topi = torch.topk(weights, k=min(5, weights.numel()))
    print(f"Destination token t={dest_t} attends to (top sources):")
    for w, i in zip(topw.tolist(), topi.tolist()):
        print(f"  source i={i} weight={w:.4f}")

    # Verify the "move information" equation: head_out[t] == sum_i attn[t,i]*V[i]
    recon = (weights.unsqueeze(-1) * V).sum(dim=0)  # [dh]
    max_abs_err = (recon - head_out[dest_t]).abs().max().item()
    print(f"\nCheck: head_out[t] == Σ_i attn[t,i]*V[i]")
    print(f"  max_abs_err = {max_abs_err:.6g}")

    # Show the residual stream "write": x_post = x_pre + delta
    x_pre = blk.last_resid_pre_attn[0, dest_t]      # [d_model]
    x_post = blk.last_resid_post_attn[0, dest_t]    # [d_model]
    delta = blk.last_attn_delta[0, dest_t]          # [d_model]
    write_err = (x_post - (x_pre + delta)).abs().max().item()
    print(f"\nCheck: residual write at t={dest_t}: x_post == x_pre + delta")
    print(f"  max_abs_err = {write_err:.6g}")


if __name__ == "__main__":
    torch.manual_seed(0)

    model = TinyDecoderOnlyTransformerLM(
        d_vocab=50,
        max_seq_len=16,
        n_layers=1,
        d_model=32,
        n_heads=1,      # make inspection simplest
        d_mlp=64,
    )

    idx = torch.tensor([[1, 7, 7, 2, 9]])  # batch=1, T=5
    inspect_first_block_attention(model, idx, dest_t=4)
