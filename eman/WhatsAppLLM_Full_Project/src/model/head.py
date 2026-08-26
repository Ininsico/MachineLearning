import torch
import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):

    def __init__(
        self,
        head_size,
        embed_dim,
        block_size,
        dropout=0.2
    ):
        super().__init__()

        self.key = nn.Linear(
            embed_dim,
            head_size,
            bias=False
        )

        self.query = nn.Linear(
            embed_dim,
            head_size,
            bias=False
        )

        self.value = nn.Linear(
            embed_dim,
            head_size,
            bias=False
        )

        self.register_buffer(
            "tril",
            torch.tril(
                torch.ones(
                    block_size,
                    block_size
                )
            )
        )

        self.dropout = nn.Dropout(
            dropout
        )

    def forward(
        self,
        x
    ):

        B, T, C = x.shape

        k = self.key(x)
        q = self.query(x)

        weights = (
            q @ k.transpose(-2, -1)
        ) * (k.shape[-1] ** -0.5)

        weights = weights.masked_fill(
            self.tril[:T, :T] == 0,
            float("-inf")
        )

        weights = F.softmax(
            weights,
            dim=-1
        )

        weights = self.dropout(
            weights
        )

        v = self.value(x)

        out = weights @ v

        return out