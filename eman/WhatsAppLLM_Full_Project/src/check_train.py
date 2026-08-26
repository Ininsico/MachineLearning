import pickle

with open("data/train.pkl", "rb") as f:
    inputs, targets = pickle.load(f)

print("Inputs:", len(inputs))
print("Targets:", len(targets))

print()

print(type(inputs[0]))
print(type(targets[0]))

print()

print("Input Length:", len(inputs[0]))
print("Target Length:", len(targets[0]))