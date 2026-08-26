"""
FLUX Text-to-Image Generation System

A production-ready text-to-image generation system powered by Ininsico.

Features:
- High-performance inference pipeline
- Advanced prompt processing
- Comprehensive evaluation metrics
- Modular architecture
- Docker & Kubernetes support
- Monitoring & logging
- Extensive test coverage

Usage:
    from src.core.pipeline import ininsicopipeline
    
    pipeline = ininsicopipeline()
    image = pipeline("a beautiful sunset", output_path="sunset.png")

API:
    uvicorn src.api:app --host 0.0.0.0 --port 8000
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__license__ = "MIT"

from .core import  ininsicopipeline
from .models import DDPMScheduler, CLIPTextEncoder, VAEEncoder, VAEDecoder
from .evaluation import FIDCalculator
from .training import PerceptualLoss, CombinedLoss

__all__ = [
    'ininsicopipeline',
    'ininsicopipeline',
    'DDPMScheduler',
    'CLIPTextEncoder',
    'VAEEncoder',
    'VAEDecoder',
    'FIDCalculator',
    'PerceptualLoss',
    'CombinedLoss'
]
