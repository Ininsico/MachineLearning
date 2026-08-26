import pickle

print("Loading dataset...")

with open(
    "data/dataset.pkl",
    "rb"
) as f:

    inputs, targets = pickle.load(f)

total = len(inputs)

train_size = int(
    total * 0.9
)

x_train = inputs[:train_size]
y_train = targets[:train_size]

x_test = inputs[train_size:]
y_test = targets[train_size:]

print(
    "Train Samples:",
    len(x_train)
)

print(
    "Test Samples:",
    len(x_test)
)

with open(
    "data/train.pkl",
    "wb"
) as f:

    pickle.dump(
        (x_train, y_train),
        f
    )

with open(
    "data/test.pkl",
    "wb"
) as f:

    pickle.dump(
        (x_test, y_test),
        f
    )

print(
    "Train/Test datasets saved."
)