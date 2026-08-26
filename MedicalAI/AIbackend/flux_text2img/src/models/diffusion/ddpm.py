import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple

class DDPMScheduler:
    """Denoising Diffusion Probabilistic Model Scheduler
    
    Implements the noise schedule from Ho et al. 2020
    https://arxiv.org/abs/2006.11239
    """
    
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
    ):
        self.num_train_timesteps = num_train_timesteps
        
        if beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        elif beta_schedule == "scaled_linear":
            self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps) ** 2
        elif beta_schedule == "squaredcos_cap_v2":
            self.betas = self._betas_for_alpha_bar(num_train_timesteps)
        
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]])
        )
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )
    
    def _betas_for_alpha_bar(self, num_diffusion_timesteps: int, max_beta: float = 0.999):
        """Create beta schedule that discretizes the given alpha_t_bar function"""
        def alpha_bar(time_step):
            return np.cos((time_step + 0.008) / 1.008 * np.pi / 2) ** 2
        
        betas = []
        for i in range(num_diffusion_timesteps):
            t1 = i / num_diffusion_timesteps
            t2 = (i + 1) / num_diffusion_timesteps
            betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
        return torch.tensor(betas)
    
    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Add noise to samples according to the noise schedule"""
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps].flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
        
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps].flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples
    
    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reverse diffusion step"""
        t = timestep
        
        pred_original_sample = (
            sample - self.sqrt_one_minus_alphas_cumprod[t] * model_output
        ) / self.sqrt_alphas_cumprod[t]
        
        pred_original_sample = torch.clamp(pred_original_sample, -1, 1)
        
        pred_prev_sample = (
            self.posterior_mean_coef1[t] * pred_original_sample +
            self.posterior_mean_coef2[t] * sample
        )
        
        variance = 0
        if t > 0:
            noise = torch.randn(
                model_output.shape,
                generator=generator,
                device=model_output.device,
                dtype=model_output.dtype,
            )
            variance = (self.posterior_variance[t] ** 0.5) * noise
        
        pred_prev_sample = pred_prev_sample + variance
        
        return pred_prev_sample, pred_original_sample
