import torch
import torch.nn as nn
import torch.nn.functional as F

class VAEEncoder(nn.Module):
    """VAE Encoder for image to latent conversion"""
    
    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 4,
        block_out_channels: tuple = (128, 256, 512, 512),
    ):
        super().__init__()
        
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], 3, padding=1)
        
        self.down_blocks = nn.ModuleList()
        in_ch = block_out_channels[0]
        for out_ch in block_out_channels:
            self.down_blocks.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 3, padding=1),
                    nn.GroupNorm(32, out_ch),
                    nn.SiLU(),
                    nn.Conv2d(out_ch, out_ch, 3, padding=1),
                    nn.GroupNorm(32, out_ch),
                    nn.SiLU(),
                    nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)
                )
            )
            in_ch = out_ch
        
        self.mid_block = nn.Sequential(
            nn.Conv2d(block_out_channels[-1], block_out_channels[-1], 3, padding=1),
            nn.GroupNorm(32, block_out_channels[-1]),
            nn.SiLU(),
            nn.Conv2d(block_out_channels[-1], block_out_channels[-1], 3, padding=1),
        )
        
        self.conv_out = nn.Conv2d(block_out_channels[-1], latent_channels * 2, 3, padding=1)
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode image to latent distribution
        
        Args:
            x: Input image (B, C, H, W)
        
        Returns:
            mean, logvar: Latent distribution parameters
        """
        h = self.conv_in(x)
        
        for down_block in self.down_blocks:
            h = down_block(h)
        
        h = self.mid_block(h)
        h = self.conv_out(h)
        
        mean, logvar = torch.chunk(h, 2, dim=1)
        
        return mean, logvar
    
    def sample(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample from latent distribution"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
