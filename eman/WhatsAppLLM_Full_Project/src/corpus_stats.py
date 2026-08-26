with open(
    "data/text_corpus.txt",
    "r",
    encoding="utf-8"
) as f:

    text = f.read()

print(
    "Characters:",
    len(text)
)

words = text.split()

print(
    "Words:",
    len(words)
)

unique_words = set(words)

print(
    "Unique Words:",
    len(unique_words)
)