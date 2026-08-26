import pickle

import torch
from torch.utils.data import Dataset


class ChatDataset(Dataset):

    def __init__(self, file_path):

        with open(file_path, "rb") as f:
            self.inputs, self.targets = pickle.load(f)

    def __len__(self):

        return len(self.inputs)

    def __getitem__(self, idx):

        return (
            torch.tensor(
                self.inputs[idx],
                dtype=torch.long
            ),
            torch.tensor(
                self.targets[idx],
                dtype=torch.long
            )
        )