import torch
import torch.nn as nn

from model.head import Head


class MultiHeadAttention(nn.Module):

    def __init__(
        self,
        num_heads,
        embed_dim,
        block_size,
        dropout=0.2
    ):
        super().__init__()

        head_size = embed_dim // num_heads

        self.heads = nn.ModuleList(
            [
                Head(
                    head_size=head_size,
                    embed_dim=embed_dim,
                    block_size=block_size,
                    dropout=dropout
                )
                for _ in range(num_heads)
            ]
        )

        self.proj = nn.Linear(
            embed_dim,
            embed_dim
        )

        self.dropout = nn.Dropout(
            dropout
        )

    def forward(
        self,
        x
    ):

        out = torch.cat(
            [head(x) for head in self.heads],
            dim=-1
        )

        out = self.proj(out)

        out = self.dropout(out)

        return out