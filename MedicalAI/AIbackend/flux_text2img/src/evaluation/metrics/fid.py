import torch
import torch.nn as nn
import numpy as np
from scipy import linalg
from typing import Tuple, List
from torchvision.models import inception_v3

class FIDCalculator:
    """Fréchet Inception Distance (FID) Calculator
    
    Measures the quality of generated images by comparing feature distributions
    from a pre-trained Inception network.
    
    Lower FID = Better quality
    Typical values: 
    - FID < 10: Excellent
    - FID 10-20: Good
    - FID 20-50: Acceptable
    - FID > 50: Poor
    """
    
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.inception_model = self._load_inception_model()
        self.inception_model.eval()
    
    def _load_inception_model(self) -> nn.Module:
        """Load pre-trained Inception V3 model"""
        model = inception_v3(pretrained=True, transform_input=False)
        model.fc = nn.Identity()  # Remove classification layer
        model = model.to(self.device)
        return model
    
    @torch.no_grad()
    def extract_features(self, images: torch.Tensor) -> np.ndarray:
        """Extract features from images using Inception network
        
        Args:
            images: Tensor of shape (N, C, H, W) in range [0, 1]
        
        Returns:
            Features of shape (N, 2048)
        """
        if images.shape[2] != 299 or images.shape[3] != 299:
            images = nn.functional.interpolate(
                images,
                size=(299, 299),
                mode='bilinear',
                align_corners=False
            )
        
        images = images.to(self.device)
        
        features = self.inception_model(images)
        
        return features.cpu().numpy()
    
    def calculate_activation_statistics(
        self,
        images: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate mean and covariance of features
        
        Args:
            images: Tensor of images
        
        Returns:
            mu: Mean of features
            sigma: Covariance matrix of features
        """
        features = self.extract_features(images)
        
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        
        return mu, sigma
    
    def calculate_frechet_distance(
        self,
        mu1: np.ndarray,
        sigma1: np.ndarray,
        mu2: np.ndarray,
        sigma2: np.ndarray,
        eps: float = 1e-6
    ) -> float:
        """Calculate Fréchet distance between two Gaussian distributions
        
        FID = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2*sqrt(sigma1*sigma2))
        
        Args:
            mu1, mu2: Mean vectors
            sigma1, sigma2: Covariance matrices
            eps: Small value for numerical stability
        
        Returns:
            FID score
        """
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)
        
        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)
        
        assert mu1.shape == mu2.shape, "Mean vectors have different lengths"
        assert sigma1.shape == sigma2.shape, "Covariance matrices have different dimensions"
        
        diff = mu1 - mu2
        
        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
        
        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError(f"Imaginary component {m}")
            covmean = covmean.real
        
        tr_covmean = np.trace(covmean)
        
        fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
        
        return float(fid)
    
    def compute_fid(
        self,
        real_images: torch.Tensor,
        generated_images: torch.Tensor
    ) -> float:
        """Compute FID score between real and generated images
        
        Args:
            real_images: Real images tensor (N, C, H, W)
            generated_images: Generated images tensor (N, C, H, W)
        
        Returns:
            FID score (lower is better)
        """
        print("Extracting features from real images...")
        mu_real, sigma_real = self.calculate_activation_statistics(real_images)
        
        print("Extracting features from generated images...")
        mu_gen, sigma_gen = self.calculate_activation_statistics(generated_images)
        
        print("Calculating FID score...")
        fid_score = self.calculate_frechet_distance(
            mu_real, sigma_real,
            mu_gen, sigma_gen
        )
        
        return fid_score
    
    def compute_fid_from_batches(
        self,
        real_batches: List[torch.Tensor],
        generated_batches: List[torch.Tensor]
    ) -> float:
        """Compute FID from list of batches (memory efficient)
        
        Args:
            real_batches: List of real image batches
            generated_batches: List of generated image batches
        
        Returns:
            FID score
        """
        real_features = []
        for batch in real_batches:
            features = self.extract_features(batch)
            real_features.append(features)
        real_features = np.concatenate(real_features, axis=0)
        
        gen_features = []
        for batch in generated_batches:
            features = self.extract_features(batch)
            gen_features.append(features)
        gen_features = np.concatenate(gen_features, axis=0)
        
        mu_real = np.mean(real_features, axis=0)
        sigma_real = np.cov(real_features, rowvar=False)
        
        mu_gen = np.mean(gen_features, axis=0)
        sigma_gen = np.cov(gen_features, rowvar=False)
        
        fid_score = self.calculate_frechet_distance(
            mu_real, sigma_real,
            mu_gen, sigma_gen
        )
        
        return fid_score

def evaluate_fid(real_images: torch.Tensor, generated_images: torch.Tensor) -> dict:
    """Convenience function to evaluate FID
    
    Returns:
        Dictionary with FID score and interpretation
    """
    calculator = FIDCalculator()
    fid = calculator.compute_fid(real_images, generated_images)
    
    if fid < 10:
        quality = "Excellent"
    elif fid < 20:
        quality = "Good"
    elif fid < 50:
        quality = "Acceptable"
    else:
        quality = "Poor"
    
    return {
        'fid_score': fid,
        'quality': quality,
        'interpretation': f"FID: {fid:.2f} ({quality})"
    }
