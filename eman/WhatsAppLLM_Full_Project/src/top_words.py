from collections import Counter

with open(
    "data/text_corpus.txt",
    "r",
    encoding="utf-8"
) as f:

    text = f.read()

words = text.split()

counter = Counter(words)

print(
    counter.most_common(50)
)