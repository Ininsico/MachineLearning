import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):

    def __init__(
        self,
        embed_dim
    ):
        super().__init__()

        self.query = nn.Linear(
            embed_dim,
            embed_dim
        )

        self.key = nn.Linear(
            embed_dim,
            embed_dim
        )

        self.value = nn.Linear(
            embed_dim,
            embed_dim
        )

    def forward(
        self,
        x
    ):

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        scores = (
            Q @ K.transpose(-2, -1)
        )

        scores = scores / (
            K.size(-1) ** 0.5
        )

        seq_len = x.size(1)

        mask = torch.tril(
            torch.ones(
                seq_len,
                seq_len,
                device=x.device
            )
        )

        scores = scores.masked_fill(
            mask == 0,
            float("-inf")
        )

        attention = F.softmax(
            scores,
            dim=-1
        )

        output = attention @ V

        return output