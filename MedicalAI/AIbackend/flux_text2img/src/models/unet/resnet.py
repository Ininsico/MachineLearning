import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNetBlock(nn.Module):
    """ResNet block with GroupNorm and optional time embedding"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int = None,
        dropout: float = 0.0,
        time_emb_channels: int = None,
        groups: int = 32,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        
        self.norm1 = nn.GroupNorm(groups, in_channels, eps=1e-6)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        
        if time_emb_channels:
            self.time_emb_proj = nn.Linear(time_emb_channels, out_channels)
        else:
            self.time_emb_proj = None
        
        self.norm2 = nn.GroupNorm(groups, out_channels, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        if in_channels != out_channels:
            self.skip_connection = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip_connection = nn.Identity()
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor = None) -> torch.Tensor:
        h = x
        
        h = self.norm1(h)
        h = F.silu(h)
        h = self.conv1(h)
        
        if time_emb is not None and self.time_emb_proj is not None:
            time_emb = F.silu(time_emb)
            time_emb = self.time_emb_proj(time_emb)[:, :, None, None]
            h = h + time_emb
        
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        return h + self.skip_connection(x)

class DownsampleBlock(nn.Module):
    """Downsample spatial dimensions by 2x"""
    
    def __init__(self, channels: int, use_conv: bool = True):
        super().__init__()
        if use_conv:
            self.op = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
        else:
            self.op = nn.AvgPool2d(2, stride=2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)

class UpsampleBlock(nn.Module):
    """Upsample spatial dimensions by 2x"""
    
    def __init__(self, channels: int, use_conv: bool = True):
        super().__init__()
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv2d(channels, channels, 3, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode='nearest')
        if self.use_conv:
            x = self.conv(x)
        return x
