"""
Custom 300M Parameter Language Model Architecture
Based on GPT-2 with custom optimizations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism"""
    
    def __init__(self, config):
        super().__init__()
        self.n_head = config['n_head']
        self.n_embd = config['n_embd']
        self.head_dim = self.n_embd // self.n_head
        
        assert self.n_embd % self.n_head == 0
        
        # Query, Key, Value projections
        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd)
        
        # Dropout
        self.attn_dropout = nn.Dropout(config['attn_pdrop'])
        self.resid_dropout = nn.Dropout(config['resid_pdrop'])
        
    def forward(self, x, mask=None):
        B, T, C = x.size()  # Batch, Sequence, Embedding
        
        # Calculate Q, K, V
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        # Reshape for multi-head attention
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        # Attention scores
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        
        if mask is not None:
            att = att.masked_fill(mask == 0, float('-inf'))
        
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        # Apply attention to values
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class FeedForward(nn.Module):
    """Position-wise feed-forward network"""
    
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config['n_embd'], config['n_inner'])
        self.c_proj = nn.Linear(config['n_inner'], config['n_embd'])
        self.dropout = nn.Dropout(config['resid_pdrop'])
        self.activation = nn.GELU()
        
    def forward(self, x):
        x = self.c_fc(x)
        x = self.activation(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Single transformer decoder block"""
    
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config['n_embd'], eps=config['layer_norm_epsilon'])
        self.attn = MultiHeadAttention(config)
        self.ln_2 = nn.LayerNorm(config['n_embd'], eps=config['layer_norm_epsilon'])
        self.mlp = FeedForward(config)
        
    def forward(self, x, mask=None):
        # Pre-norm architecture
        x = x + self.attn(self.ln_1(x), mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class CustomLM(nn.Module):
    """
    Custom Language Model with 300M parameters
    
    Architecture:
    - Token + Position Embeddings
    - 24 Transformer Blocks
    - Layer Norm
    - Language Modeling Head
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.wte = nn.Embedding(config['vocab_size'], config['n_embd'])
        self.wpe = nn.Embedding(config['n_positions'], config['n_embd'])
        self.drop = nn.Dropout(config['embd_pdrop'])
        
        # Transformer blocks
        self.h = nn.ModuleList([
            TransformerBlock(config) for _ in range(config['n_layer'])
        ])
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(config['n_embd'], eps=config['layer_norm_epsilon'])
        
        # Language modeling head
        self.lm_head = nn.Linear(config['n_embd'], config['vocab_size'], bias=False)
        
        # Weight tying
        self.wte.weight = self.lm_head.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Calculate total parameters
        self.total_params = sum(p.numel() for p in self.parameters())
        print(f"Model initialized with {self.total_params:,} parameters")
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config['initializer_range'])
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config['initializer_range'])
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, input_ids, labels=None):
        device = input_ids.device
        b, t = input_ids.size()
        
        assert t <= self.config['n_positions'], f"Sequence length {t} exceeds maximum {self.config['n_positions']}"
        
        # Token embeddings
        tok_emb = self.wte(input_ids)
        
        # Position embeddings
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)
        pos_emb = self.wpe(pos)
        
        # Combine embeddings
        x = self.drop(tok_emb + pos_emb)
        
        # Create causal mask
        mask = torch.tril(torch.ones(t, t, device=device)).view(1, 1, t, t)
        
        # Apply transformer blocks
        for block in self.h:
            x = block(x, mask)
        
        # Final layer norm
        x = self.ln_f(x)
        
        # Language modeling head
        logits = self.lm_head(x)
        
        loss = None
        if labels is not None:
            # Shift logits and labels for next token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Calculate loss
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )
        
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Generate text autoregressively"""
        for _ in range(max_new_tokens):
            # Crop context if needed
            idx_cond = idx if idx.size(1) <= self.config['n_positions'] else idx[:, -self.config['n_positions']:]
            
            # Forward pass
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            # Top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx
    
    @classmethod
    def from_pretrained(cls, checkpoint_path):
        """Load model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        config = checkpoint['config']
        model = cls(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
    
    def get_num_params(self):
        """Return number of parameters"""
        return self.total_params


if __name__ == "__main__":
    # Test model initialization
    config = {
        'vocab_size': 50257,
        'n_positions': 2048,
        'n_embd': 1024,
        'n_layer': 24,
        'n_head': 16,
        'n_inner': 4096,
        'embd_pdrop': 0.1,
        'resid_pdrop': 0.1,
        'attn_pdrop': 0.1,
        'layer_norm_epsilon': 1e-5,
        'initializer_range': 0.02
    }
    
    model = CustomLM(config)
    print(f"\nModel Architecture:")
    print(f"Total Parameters: {model.get_num_params():,}")
    print(f"Layers: {config['n_layer']}")
    print(f"Hidden Size: {config['n_embd']}")
    print(f"Attention Heads: {config['n_head']}")
