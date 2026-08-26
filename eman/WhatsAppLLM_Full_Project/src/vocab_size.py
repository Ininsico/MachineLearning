from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file(
    "models/tokenizer.json"
)

print(
    "Vocabulary Size:",
    tokenizer.get_vocab_size()
)