import torch
import torch.nn as nn


class LayerNorm(nn.Module):

    def __init__(
        self,
        embed_dim
    ):
        super().__init__()

        self.norm = nn.LayerNorm(
            embed_dim
        )

    def forward(
        self,
        x
    ):
        return self.norm(x)