import json

from tokenizers import Tokenizer

print("Loading tokenizer...")

tokenizer = Tokenizer.from_file(
    "models/tokenizer.json"
)

print("Loading corpus...")

with open(
    "data/text_corpus.txt",
    "r",
    encoding="utf-8"
) as f:

    text = f.read()

print("Encoding corpus...")

text = "[BOS] " + text + " [EOS]"

encoded = tokenizer.encode(
    text
)

token_ids = encoded.ids

print(
    f"Total Tokens: {len(token_ids)}"
)

with open(
    "data/token_ids.json",
    "w"
) as f:

    json.dump(
        token_ids,
        f
    )

print(
    "Token IDs saved successfully."
)