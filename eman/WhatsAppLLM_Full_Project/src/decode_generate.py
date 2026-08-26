import torch

from tokenizers import Tokenizer

from model.gpt import GPTLanguageModel


VOCAB_SIZE = 8000


tokenizer = Tokenizer.from_file(
    "models/tokenizer.json"
)


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


generated = model.generate(
    context,
    max_new_tokens=50
)


tokens = generated[0].tolist()

text = tokenizer.decode(
    tokens
)

print("\nGenerated Text:\n")
print(text)