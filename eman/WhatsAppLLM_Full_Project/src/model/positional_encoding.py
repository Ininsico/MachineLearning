import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):

    def __init__(
        self,
        max_len,
        embed_dim
    ):
        super().__init__()

        self.position_embedding = nn.Embedding(
            max_len,
            embed_dim
        )

    def forward(
        self,
        x
    ):
        batch_size, seq_len, embed_dim = x.shape

        positions = torch.arange(
            seq_len,
            device=x.device
        )

        positions = positions.unsqueeze(0)

        position_embeddings = (
            self.position_embedding(
                positions
            )
        )

        return x + position_embeddings