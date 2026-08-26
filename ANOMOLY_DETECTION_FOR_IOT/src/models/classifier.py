import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.block(x))

class AnomalyClassifier(nn.Module):

    def __init__(self, input_dim: int, hidden_dims: list, dropout: float = 0.3):
        super().__init__()

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers += [
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
            prev_dim = h_dim

        layers.append(ResidualBlock(prev_dim, dropout))

        layers += [
            nn.Linear(prev_dim, 1),
            nn.Sigmoid(),
        ]
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(1)

def build_classifier(config, n_features: int, device: torch.device) -> AnomalyClassifier:
    model = AnomalyClassifier(
        input_dim=n_features,
        hidden_dims=config.CLF_HIDDEN,
        dropout=config.CLF_DROPOUT,
    ).to(device)
    return model