import torch
import torch.nn as nn
from transformers import CLIPTextModel, CLIPTokenizer

class CLIPTextEncoder(nn.Module):
    """CLIP text encoder for conditioning"""
    
    def __init__(self, model_name: str = "openai/clip-vit-large-patch14"):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.text_encoder = CLIPTextModel.from_pretrained(model_name)
        self.text_encoder.eval()
        
        for param in self.text_encoder.parameters():
            param.requires_grad = False
    
    @torch.no_grad()
    def encode(self, prompts: list[str], device: str = "cuda") -> torch.Tensor:
        """Encode text prompts to embeddings
        
        Args:
            prompts: List of text prompts
            device: Device to use
        
        Returns:
            Text embeddings of shape (batch_size, seq_len, hidden_dim)
        """
        inputs = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt"
        )
        
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        outputs = self.text_encoder(**inputs)
        embeddings = outputs.last_hidden_state
        
        return embeddings
    
    def __call__(self, prompts: list[str], device: str = "cuda") -> torch.Tensor:
        return self.encode(prompts, device)
