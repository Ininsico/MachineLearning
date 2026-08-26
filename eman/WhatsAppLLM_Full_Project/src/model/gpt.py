import torch
import torch.nn as nn

from model.block import Block


class GPTLanguageModel(nn.Module):

    def __init__(
        self,
        vocab_size,
        embed_dim=128,
        block_size=128,
        num_heads=4,
        num_layers=2
    ):
        super().__init__()

        self.token_embedding_table = nn.Embedding(
            vocab_size,
            embed_dim
        )

        self.position_embedding_table = nn.Embedding(
            block_size,
            embed_dim
        )

        self.blocks = nn.Sequential(
            *[
                Block(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    block_size=block_size
                )
                for _ in range(num_layers)
            ]
        )

        self.final_layer_norm = nn.LayerNorm(
            embed_dim
        )

        self.final_linear_layer = nn.Linear(
            embed_dim,
            vocab_size
        )

    def forward(
        self,
        idx
    ):

        B, T = idx.shape

        token_embeddings = self.token_embedding_table(
            idx
        )

        positions = torch.arange(
            T,
            device=idx.device
        )

        position_embeddings = (
            self.position_embedding_table(
                positions
            )
        )

        x = token_embeddings + position_embeddings

        x = self.blocks(x)

        x = self.final_layer_norm(x)

        logits = self.final_linear_layer(x)

        return logits

    def generate(
        self,
        idx,
        max_new_tokens
    ):

        for _ in range(max_new_tokens):

            logits = self(idx)

            logits = logits[:, -1, :]

            probs = torch.softmax(
                logits,
                dim=-1
            )

            next_token = torch.multinomial(
                probs,
                num_samples=1
            )

            idx = torch.cat(
                (
                    idx,
                    next_token
                ),
                dim=1
            )

        return idx