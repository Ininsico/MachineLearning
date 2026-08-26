from torch.utils.data import DataLoader

from dataloader import ChatDataset


BATCH_SIZE = 8


train_dataset = ChatDataset(
    "data/train.pkl"
)

test_dataset = ChatDataset(
    "data/test.pkl"
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print(
    "Train Batches:",
    len(train_loader)
)

print(
    "Test Batches:",
    len(test_loader)
)

x, y = next(
    iter(train_loader)
)

print(
    "Input Batch Shape:",
    x.shape
)

print(
    "Target Batch Shape:",
    y.shape
)