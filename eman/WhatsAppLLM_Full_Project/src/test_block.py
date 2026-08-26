import torch

from model.block import Block

block = Block(
    embed_dim=384,
    num_heads=6,
    block_size=256
)

x = torch.randn(
    2,
    10,
    384
)

out = block(x)

print(out.shape)