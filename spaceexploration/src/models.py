import torch
import torch.nn as nn

from src import config


class NEOMLP(nn.Module):
    """Multi-layer perceptron for hazardous-NEO classification."""

    def __init__(self, n_features, hidden_dims=config.HIDDEN_DIMS, dropout=config.DROPOUT):
        super().__init__()
        layers = []
        in_dim = n_features
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)
