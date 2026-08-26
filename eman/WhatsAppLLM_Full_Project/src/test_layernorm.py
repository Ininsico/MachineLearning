import torch

from model.layernorm import LayerNorm

layer_norm = LayerNorm(
    256
)

x = torch.randn(
    1,
    3,
    256
)

output = layer_norm(x)

print(output.shape)