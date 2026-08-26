import pickle

with open(
    "data/dataset.pkl",
    "rb"
) as f:

    inputs, targets = pickle.load(f)

print(
    "Samples:",
    len(inputs)
)

print(
    "\nInput Length:",
    len(inputs[0])
)

print(
    "\nTarget Length:",
    len(targets[0])
)

print(
    "\nFirst Input:"
)

print(
    inputs[0][:20]
)

print(
    "\nFirst Target:"
)

print(
    targets[0][:20]
)