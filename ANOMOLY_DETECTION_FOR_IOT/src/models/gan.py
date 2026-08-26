import torch
import torch.nn as nn

def weights_init(m):
    if isinstance(m, (nn.Linear,)):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class Generator(nn.Module):

    def __init__(self, latent_dim: int, output_dim: int):
        super().__init__()
        self.model = nn.Sequential(

            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(512, output_dim),
            nn.Tanh(),
        )
        self.apply(weights_init)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.model(z)

class Discriminator(nn.Module):

    def __init__(self, input_dim: int):
        super().__init__()
        self.model = nn.Sequential(

            nn.utils.spectral_norm(nn.Linear(input_dim, 512)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.utils.spectral_norm(nn.Linear(512, 256)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.utils.spectral_norm(nn.Linear(256, 128)),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
        self.apply(weights_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

def build_gan(config, n_features: int, device: torch.device):
    G = Generator(config.LATENT_DIM, n_features).to(device)
    D = Discriminator(n_features).to(device)
    return G, D