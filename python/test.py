# import  torch
# from torch import nn
# import torch.nn.init as init
#
# num_of_gpus = torch.cuda.device_count()
# print(f"num_of_gpus={num_of_gpus}")
#
# is_available = torch.cuda.is_available()
# print(f"is_available={is_available}")
#
# #two tokens, embedding length 3
# x = torch.tensor([1, 1, 1, 2, 2, 2], dtype=torch.float32).view(2, 3)
#
#
# print(f"x\n={x}")
#
# d_model = 3
#
# qkv = nn.Linear(d_model, 3 * d_model, bias=False)
# print(f"qkv.weight.size={qkv.weight.size()}")
# # qkv_weight = torch.tensor([1,2,3], dtype=torch.float32).view(3,1).broadcast_to(3, 9).transpose(1,0)
# qkv_weight = torch.tensor(
#     [
#         1,2,3,  10,20,30,   100,200,300,
#         1,2,3,  10,20,30,   100,200,300,
#         1,2,3,  10,20,30,   100,200,300,
#      ], dtype=torch.float32).view(3,9).transpose(1,0)
# print(f"qkv_weight=\n{qkv_weight}")
# q, k, v = qkv_weight.chunk(3, dim=-1)
#
# print(f"q=\n{q}")
# print(f"k=\n{k}")
# print(f"v=\n{v}")
#
# with torch.no_grad():
#     qkv.weight.copy_(qkv_weight)
#
# print(f"qkv.weight={qkv.weight} qkv.bias={qkv.bias}")
#
# qkv_x = qkv(x)
#
# print(f"qkv_x\n={qkv_x}")
#
# q_x, k_x, v_x = qkv_x.chunk(3, dim=-1)
#
# print(f"q_x=\n{q_x}")
# print(f"k_x=\n{k_x}")
# print(f"v_x=\n{v_x}")
#
# print("Ok")

import torch
import math

torch.manual_seed(0)

# ---- dimensions ----
B, L = 2, 4          # batch, sequence length
D = 8                # d_model
H = 2                # num_heads
Dh = D // H          # d_head

# ---- fake input + weights ----
x   = torch.randn(B, L, D)

W_Q = torch.randn(D, D)
W_K = torch.randn(D, D)
W_V = torch.randn(D, D)
W_O = torch.randn(D, D)   # output projection

# ---- compute Q,K,V then split into heads ----
q = x @ W_Q   # [B,L,D]
k = x @ W_K
v = x @ W_V

# [B,L,D] -> [B,H,L,Dh]
q = q.view(B, L, H, Dh).transpose(1, 2)
k = k.view(B, L, H, Dh).transpose(1, 2)
v = v.view(B, L, H, Dh).transpose(1, 2)

# ---- per-head attention (independent heads) ----
scores = (q @ k.transpose(-2, -1)) / math.sqrt(Dh)    # [B,H,L,L]
attn   = torch.softmax(scores, dim=-1)               # [B,H,L,L]
head   = attn @ v                                    # [B,H,L,Dh]

# reshape to [B,L,H,Dh] for convenience
head_blhd = head.transpose(1, 2)                      # [B,L,H,Dh]

# ============================================================
# 1) "Standard" MHA: concat heads then apply a single W_O
# ============================================================
concat = head_blhd.reshape(B, L, D)                   # [B,L,D]
y_standard = concat @ W_O                             # [B,L,D]
lstm
# ============================================================
# 2) "Independent + additive":
#    Split W_O into H blocks, one per head-chunk of size Dh,
#    apply each block to its head output, and SUM.
# ============================================================
# W_O maps from D -> D. Since concat is [head0 | head1 | ...],
# we split W_O by INPUT rows into H chunks of shape [Dh, D].
W_O_blocks = W_O.split(Dh, dim=0)                     # tuple of H tensors, each [Dh, D]

y_additive = torch.zeros(B, L, D)

for h in range(H):
    head_h = head_blhd[:, :, h, :]                    # [B,L,Dh]
    # head_h @ W_O_blocks[h] gives [B,L,D]
    y_additive += head_h @ W_O_blocks[h]

print("max |y_standard - y_additive| =", (y_standard - y_additive).abs().max().item())

# ============================================================
# 3) Show additivity: remove one head's contribution explicitly
# ============================================================
h = 0
contrib_h = head_blhd[:, :, h, :] @ W_O_blocks[h]     # [B,L,D]
y_without_h = y_additive - contrib_h

# Recompute by zeroing that head and doing the same loop
head_zeroed = head_blhd.clone()
head_zeroed[:, :, h, :] = 0

y_recomputed = torch.zeros(B, L, D)
for hh in range(H):
    y_recomputed += head_zeroed[:, :, hh, :] @ W_O_blocks[hh]

print("max |(y_without_h) - (y_recomputed)| =", (y_without_h - y_recomputed).abs().max().item())
