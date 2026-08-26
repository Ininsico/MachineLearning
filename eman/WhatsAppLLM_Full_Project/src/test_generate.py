import torch

from model.gpt import GPTLanguageModel


VOCAB_SIZE = 8000


model = GPTLanguageModel(
    vocab_size=VOCAB_SIZE
)

model.load_state_dict(
    torch.load(
        "models/gpt_model.pth",
        map_location="cpu"
    )
)

model.eval()


context = torch.tensor(
    [[2]]
)

output = model.generate(
    context,
    max_new_tokens=20
)

print(output)