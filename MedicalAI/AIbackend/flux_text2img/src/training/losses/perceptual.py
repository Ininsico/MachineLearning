import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights
from typing import List, Optional

class PerceptualLoss(nn.Module):
    """Perceptual Loss using VGG16 features
    
    Compares high-level features instead of pixel values.
    Better for image quality assessment than MSE/L1.
    
    Used in:
    - Style transfer
    - Super-resolution
    - Image-to-image translation
    - Diffusion model training
    """
    
    def __init__(
        self,
        layers: List[str] = ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'],
        weights: Optional[List[float]] = None,
        normalize: bool = True,
    ):
        super().__init__()
        
        self.layers = layers
        self.weights = weights or [1.0] * len(layers)
        self.normalize = normalize
        
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        vgg.eval()
        
        for param in vgg.parameters():
            param.requires_grad = False
        
        self.vgg_layers = nn.ModuleDict()
        
        layer_mapping = {
            'relu1_1': 1, 'relu1_2': 3,
            'relu2_1': 6, 'relu2_2': 8,
            'relu3_1': 11, 'relu3_2': 13, 'relu3_3': 15,
            'relu4_1': 18, 'relu4_2': 20, 'relu4_3': 22,
            'relu5_1': 25, 'relu5_2': 27, 'relu5_3': 29,
        }
        
        for layer_name in layers:
            if layer_name not in layer_mapping:
                raise ValueError(f"Unknown layer: {layer_name}")
            
            layer_idx = layer_mapping[layer_name]
            self.vgg_layers[layer_name] = nn.Sequential(*list(vgg.children())[:layer_idx+1])
        
        if normalize:
            self.register_buffer(
                'mean',
                torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            )
            self.register_buffer(
                'std',
                torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            )
    
    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize to ImageNet stats"""
        if self.normalize:
            return (x - self.mean) / self.std
        return x
    
    def extract_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract features from all specified layers"""
        x = self._normalize(x)
        
        features = []
        for layer_name in self.layers:
            x_feat = self.vgg_layers[layer_name](x)
            features.append(x_feat)
        
        return features
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        reduction: str = 'mean'
    ) -> torch.Tensor:
        """Compute perceptual loss
        
        Args:
            pred: Predicted images (B, 3, H, W) in range [0, 1]
            target: Target images (B, 3, H, W) in range [0, 1]
            reduction: 'mean', 'sum', or 'none'
        
        Returns:
            Perceptual loss value
        """
        if pred.shape[1] == 1:
            pred = pred.repeat(1, 3, 1, 1)
        if target.shape[1] == 1:
            target = target.repeat(1, 3, 1, 1)
        
        pred_features = self.extract_features(pred)
        target_features = self.extract_features(target)
        
        loss = 0.0
        for pred_feat, target_feat, weight in zip(pred_features, target_features, self.weights):
            if reduction == 'mean':
                loss += weight * F.mse_loss(pred_feat, target_feat)
            elif reduction == 'sum':
                loss += weight * F.mse_loss(pred_feat, target_feat, reduction='sum')
            else:
                loss += weight * F.mse_loss(pred_feat, target_feat, reduction='none')
        
        return loss

class StyleLoss(nn.Module):
    """Style Loss using Gram matrices
    
    Captures texture and style information.
    Used in neural style transfer.
    """
    
    def __init__(
        self,
        layers: List[str] = ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'],
        weights: Optional[List[float]] = None,
    ):
        super().__init__()
        self.perceptual = PerceptualLoss(layers=layers, weights=weights)
    
    def gram_matrix(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Gram matrix for style representation"""
        b, c, h, w = x.size()
        features = x.view(b, c, h * w)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram / (c * h * w)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute style loss using Gram matrices"""
        pred_features = self.perceptual.extract_features(pred)
        target_features = self.perceptual.extract_features(target)
        
        loss = 0.0
        for pred_feat, target_feat, weight in zip(
            pred_features, target_features, self.perceptual.weights
        ):
            pred_gram = self.gram_matrix(pred_feat)
            target_gram = self.gram_matrix(target_feat)
            loss += weight * F.mse_loss(pred_gram, target_gram)
        
        return loss

class CombinedLoss(nn.Module):
    """Combined loss: Pixel + Perceptual + Style
    
    Balances:
    - Pixel-level accuracy (L1/L2)
    - Perceptual quality (VGG features)
    - Style consistency (Gram matrices)
    """
    
    def __init__(
        self,
        pixel_weight: float = 1.0,
        perceptual_weight: float = 1.0,
        style_weight: float = 0.0,
        pixel_loss: str = 'l1',  # 'l1' or 'l2'
    ):
        super().__init__()
        
        self.pixel_weight = pixel_weight
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        
        if pixel_loss == 'l1':
            self.pixel_loss = nn.L1Loss()
        elif pixel_loss == 'l2':
            self.pixel_loss = nn.MSELoss()
        else:
            raise ValueError(f"Unknown pixel loss: {pixel_loss}")
        
        if perceptual_weight > 0:
            self.perceptual_loss = PerceptualLoss()
        
        if style_weight > 0:
            self.style_loss = StyleLoss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict:
        """Compute combined loss
        
        Returns:
            Dictionary with individual and total losses
        """
        losses = {}
        
        if self.pixel_weight > 0:
            losses['pixel'] = self.pixel_loss(pred, target) * self.pixel_weight
        
        if self.perceptual_weight > 0:
            losses['perceptual'] = self.perceptual_loss(pred, target) * self.perceptual_weight
        
        if self.style_weight > 0:
            losses['style'] = self.style_loss(pred, target) * self.style_weight
        
        losses['total'] = sum(losses.values())
        
        return losses

# Example usage
def example_usage():
    """Example of using perceptual loss in training"""
    
    # Initialize loss
    criterion = CombinedLoss(
        pixel_weight=1.0,
        perceptual_weight=0.1,
        style_weight=0.01,
        pixel_loss='l1'
    )
    
    # Dummy data
    pred = torch.rand(4, 3, 256, 256)  # Generated images
    target = torch.rand(4, 3, 256, 256)  # Ground truth
    
    # Compute loss
    losses = criterion(pred, target)
    
    print(f"Pixel Loss: {losses['pixel']:.4f}")
    print(f"Perceptual Loss: {losses['perceptual']:.4f}")
    if 'style' in losses:
        print(f"Style Loss: {losses['style']:.4f}")
    print(f"Total Loss: {losses['total']:.4f}")
    
    # Backprop
    losses['total'].backward()

if __name__ == "__main__":
    example_usage()
