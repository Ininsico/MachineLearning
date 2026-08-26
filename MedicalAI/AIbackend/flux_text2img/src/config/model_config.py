from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    """Configuration for FLUX model"""
    
    model_id: str = "black-forest-labs/FLUX.1-schnell"
    
    # Image generation
    default_width: int = 1024
    default_height: int = 1024
    min_size: int = 256
    max_size: int = 2048
    
    # Inference
    default_steps: int = 4
    min_steps: int = 1
    max_steps: int = 50
    
    default_guidance: float = 3.5
    min_guidance: float = 1.0
    max_guidance: float = 20.0
    
    # Performance
    use_fp16: bool = True
    use_attention_slicing: bool = True
    use_cpu_offload: bool = False
    
    # Safety
    enable_safety_checker: bool = True
    enable_watermark: bool = False
    
    # Caching
    cache_dir: Optional[str] = None
    use_model_cache: bool = True

@dataclass
class UNetConfig:
    """UNet architecture configuration"""
    
    in_channels: int = 4
    out_channels: int = 4
    model_channels: int = 320
    
    attention_resolutions: tuple = (4, 2, 1)
    num_res_blocks: int = 2
    channel_mult: tuple = (1, 2, 4, 4)
    
    num_heads: int = 8
    num_head_channels: int = 64
    
    use_spatial_transformer: bool = True
    transformer_depth: int = 1
    context_dim: int = 768
    
    dropout: float = 0.0
    use_checkpoint: bool = False

@dataclass
class VAEConfig:
    """VAE configuration"""
    
    in_channels: int = 3
    out_channels: int = 3
    latent_channels: int = 4
    
    down_block_types: tuple = ("DownEncoderBlock2D",) * 4
    up_block_types: tuple = ("UpDecoderBlock2D",) * 4
    
    block_out_channels: tuple = (128, 256, 512, 512)
    layers_per_block: int = 2
    
    scaling_factor: float = 0.18215

@dataclass
class SchedulerConfig:
    """Diffusion scheduler configuration"""
    
    num_train_timesteps: int = 1000
    beta_start: float = 0.00085
    beta_end: float = 0.012
    beta_schedule: str = "scaled_linear"
    
    prediction_type: str = "epsilon"
    timestep_spacing: str = "leading"
    steps_offset: int = 1

# Default configurations
DEFAULT_MODEL_CONFIG = ModelConfig()
DEFAULT_UNET_CONFIG = UNetConfig()
DEFAULT_VAE_CONFIG = VAEConfig()
DEFAULT_SCHEDULER_CONFIG = SchedulerConfig()
