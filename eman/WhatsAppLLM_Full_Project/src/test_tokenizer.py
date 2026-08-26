from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file(
    "models/tokenizer.json"
)

text = "laptop password project"

encoded = tokenizer.encode(
    text
)

print("\nTokens:")
print(encoded.tokens)

print("\nIDs:")
print(encoded.ids)

decoded = tokenizer.decode(
    encoded.ids
)

print("\nDecoded:")
print(decoded)