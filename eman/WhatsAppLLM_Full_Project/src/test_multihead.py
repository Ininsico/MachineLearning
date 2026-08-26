import torch

from model.multihead_attention import MultiHeadAttention

mha = MultiHeadAttention(
    num_heads=6,
    embed_dim=384,
    block_size=256
)

x = torch.randn(
    2,
    10,
    384
)

out = mha(x)

print(out.shape)