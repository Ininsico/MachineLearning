import torch

from model.head import Head

head = Head(
    head_size=64,
    embed_dim=384,
    block_size=256
)

x = torch.randn(
    2,
    10,
    384
)

out = head(x)

print(out.shape)