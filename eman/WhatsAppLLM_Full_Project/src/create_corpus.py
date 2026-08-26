import pandas as pd

print("Loading cleaned dataset...")

df = pd.read_csv(
    "data/cleaned_chat.csv"
)

print(
    f"Messages Loaded: {len(df)}"
)

text_corpus = " ".join(
    df["message"].astype(str)
)

with open(
    "data/text_corpus.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write(text_corpus)

print(
    "\nCorpus Created Successfully!"
)

print(
    f"Corpus Length: {len(text_corpus)} characters"
)