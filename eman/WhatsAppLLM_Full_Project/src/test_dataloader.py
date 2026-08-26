from dataloader import ChatDataset

dataset = ChatDataset(
    "data/train.pkl"
)

print(
    "Samples:",
    len(dataset)
)

x, y = dataset[0]

print(
    "Input Shape:",
    x.shape
)

print(
    "Target Shape:",
    y.shape
)