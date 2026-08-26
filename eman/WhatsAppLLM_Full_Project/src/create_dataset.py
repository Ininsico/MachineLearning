import json
import pickle

SEQ_LEN = 128

print("Loading token IDs...")

with open(
    "data/token_ids.json",
    "r"
) as f:
    token_ids = json.load(f)

print(
    "Total Tokens:",
    len(token_ids)
)

inputs = []
targets = []

for i in range(
    len(token_ids) - SEQ_LEN
):

    x = token_ids[
        i:i+SEQ_LEN
    ]

    y = token_ids[
        i+1:i+SEQ_LEN+1
    ]

    inputs.append(x)
    targets.append(y)

print(
    "Training Samples:",
    len(inputs)
)

with open(
    "data/dataset.pkl",
    "wb"
) as f:

    pickle.dump(
        (inputs, targets),
        f
    )

print(
    "Dataset saved."
)