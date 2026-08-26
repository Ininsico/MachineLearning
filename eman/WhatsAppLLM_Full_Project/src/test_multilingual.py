from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file(
    "models/tokenizer.json"
)

text = "mujhe laptop project acha laga"

encoded = tokenizer.encode(text)

print("Tokens:")
print(encoded.tokens)

print("\nIDs:")
print(encoded.ids)