import torch

from model.gpt import GPTLanguageModel

VOCAB_SIZE = 7000

model = GPTLanguageModel(
    vocab_size=VOCAB_SIZE
)

x = torch.randint(
    0,
    VOCAB_SIZE,
    (2, 10)
)

out = model(x)

print(out.shape)