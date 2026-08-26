from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file(
    "models/tokenizer.json"
)

vocab = tokenizer.get_vocab()

sorted_vocab = sorted(
    vocab.items(),
    key=lambda x: x[1]
)

for token, token_id in sorted_vocab[:200]:
    print(token_id, "->", token)