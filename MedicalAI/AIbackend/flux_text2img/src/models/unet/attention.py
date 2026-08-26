import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention with Flash Attention optimization"""
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x

class CrossAttention(nn.Module):
    """Cross-Attention for conditioning on text embeddings"""
    
    def __init__(
        self,
        query_dim: int,
        context_dim: Optional[int] = None,
        num_heads: int = 8,
        head_dim: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        inner_dim = head_dim * num_heads
        context_dim = context_dim or query_dim
        
        self.scale = head_dim ** -0.5
        self.num_heads = num_heads
        
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.num_heads
        
        q = self.to_q(x)
        context = context if context is not None else x
        k = self.to_k(context)
        v = self.to_v(context)
        
        q, k, v = map(lambda t: t.reshape(t.shape[0], t.shape[1], h, -1).transpose(1, 2), (q, k, v))
        
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        
        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(out.shape[0], out.shape[2], -1)
        
        return self.to_out(out)

class SpatialTransformer(nn.Module):
    """Spatial Transformer block with self and cross attention"""
    
    def __init__(
        self,
        in_channels: int,
        num_heads: int,
        head_dim: int,
        context_dim: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels, eps=1e-6)
        self.attn1 = MultiHeadAttention(
            in_channels,
            num_heads=num_heads,
            attn_drop=dropout,
            proj_drop=dropout,
        )
        
        self.norm2 = nn.GroupNorm(32, in_channels, eps=1e-6)
        self.attn2 = CrossAttention(
            query_dim=in_channels,
            context_dim=context_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
        )
        
        self.norm3 = nn.GroupNorm(32, in_channels, eps=1e-6)
        self.ff = FeedForward(in_channels, dropout=dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, c, h, w = x.shape
        x_in = x
        
        x = self.norm1(x)
        x = x.reshape(b, c, h * w).transpose(1, 2)
        x = self.attn1(x) + x
        
        x = x.transpose(1, 2).reshape(b, c, h, w)
        x = self.norm2(x)
        x = x.reshape(b, c, h * w).transpose(1, 2)
        x = self.attn2(x, context=context) + x
        
        x = x.transpose(1, 2).reshape(b, c, h, w)
        x = self.norm3(x)
        x = x.reshape(b, c, h * w).transpose(1, 2)
        x = self.ff(x) + x
        
        x = x.transpose(1, 2).reshape(b, c, h, w)
        
        return x + x_in

class FeedForward(nn.Module):
    """Position-wise Feed-Forward Network with GELU activation"""
    
    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
