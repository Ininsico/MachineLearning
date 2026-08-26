import torch

from model.embedding import TokenEmbedding
from model.positional_encoding import PositionalEncoding
from model.attention import SelfAttention
from model.feedforward import FeedForward

VOCAB_SIZE = 8000
EMBED_DIM = 256
HIDDEN_DIM = 1024
MAX_LEN = 128

embedding = TokenEmbedding(
    VOCAB_SIZE,
    EMBED_DIM
)

position = PositionalEncoding(
    MAX_LEN,
    EMBED_DIM
)

attention = SelfAttention(
    EMBED_DIM
)

feedforward = FeedForward(
    EMBED_DIM,
    HIDDEN_DIM
)

sample = torch.tensor([
    [4115, 1062, 891]
])

x = embedding(sample)

x = position(x)

x = attention(x)

output = feedforward(x)

print(
    output.shape
)