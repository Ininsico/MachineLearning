import torch

from model.embedding import TokenEmbedding

VOCAB_SIZE = 8000
EMBED_DIM = 256

embedding = TokenEmbedding(
    VOCAB_SIZE,
    EMBED_DIM
)

sample = torch.tensor([
    [4115,1062,891]
])

output = embedding(sample)

print(
    output.shape
)