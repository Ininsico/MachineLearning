import torch
import torch.nn as nn

class VAEDecoder(nn.Module):
    """VAE Decoder for latent to image conversion"""
    
    def __init__(
        self,
        latent_channels: int = 4,
        out_channels: int = 3,
        block_out_channels: tuple = (512, 512, 256, 128),
    ):
        super().__init__()
        
        self.conv_in = nn.Conv2d(latent_channels, block_out_channels[0], 3, padding=1)
        
        self.mid_block = nn.Sequential(
            nn.Conv2d(block_out_channels[0], block_out_channels[0], 3, padding=1),
            nn.GroupNorm(32, block_out_channels[0]),
            nn.SiLU(),
            nn.Conv2d(block_out_channels[0], block_out_channels[0], 3, padding=1),
        )
        
        self.up_blocks = nn.ModuleList()
        in_ch = block_out_channels[0]
        for out_ch in block_out_channels:
            self.up_blocks.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 3, padding=1),
                    nn.GroupNorm(32, out_ch),
                    nn.SiLU(),
                    nn.Conv2d(out_ch, out_ch, 3, padding=1),
                    nn.GroupNorm(32, out_ch),
                    nn.SiLU(),
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv2d(out_ch, out_ch, 3, padding=1)
                )
            )
            in_ch = out_ch
        
        self.conv_out = nn.Conv2d(block_out_channels[-1], out_channels, 3, padding=1)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to image
        
        Args:
            z: Latent tensor (B, latent_channels, H, W)
        
        Returns:
            Decoded image (B, out_channels, H*8, W*8)
        """
        h = self.conv_in(z)
        h = self.mid_block(h)
        
        for up_block in self.up_blocks:
            h = up_block(h)
        
        h = self.conv_out(h)
        
        return torch.tanh(h)
