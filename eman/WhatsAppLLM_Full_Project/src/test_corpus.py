with open(
    "data/text_corpus.txt",
    "r",
    encoding="utf-8"
) as f:

    corpus = f.read()

print(
    corpus[:1000]
)

print(
    "\nTotal Characters:",
    len(corpus)
)