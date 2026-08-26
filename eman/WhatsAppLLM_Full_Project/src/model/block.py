import torch
import torch.nn as nn

from model.multihead_attention import MultiHeadAttention
from model.feedforward import FeedForward


class Block(nn.Module):

    def __init__(
        self,
        embed_dim,
        num_heads,
        block_size,
        dropout=0.2
    ):
        super().__init__()

        self.sa = MultiHeadAttention(
            num_heads=num_heads,
            embed_dim=embed_dim,
            block_size=block_size,
            dropout=dropout
        )

        self.ffwd = FeedForward(
            embed_dim=embed_dim,
            hidden_dim=4 * embed_dim
        )

        self.ln1 = nn.LayerNorm(
            embed_dim
        )

        self.ln2 = nn.LayerNorm(
            embed_dim
        )

    def forward(
        self,
        x
    ):

        x = x + self.sa(
            self.ln1(x)
        )

        x = x + self.ffwd(
            self.ln2(x)
        )

        return x