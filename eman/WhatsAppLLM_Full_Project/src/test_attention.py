import torch

from model.embedding import TokenEmbedding
from model.positional_encoding import PositionalEncoding
from model.attention import SelfAttention

VOCAB_SIZE = 8000
EMBED_DIM = 256
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

sample = torch.tensor([
    [4115, 1062, 891]
])

x = embedding(sample)

x = position(x)

output = attention(x)

print(
    output.shape
)