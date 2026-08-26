import torch
from typing import List, Optional
from pathlib import Path
from tqdm import tqdm

from ...core.pipeline import FluxPipeline
from ...utils.logging import get_logger

logger = get_logger(__name__)

class BatchProcessor:
    """Process multiple prompts in batches for efficient generation"""
    
    def __init__(self, batch_size: int = 4, num_workers: int = 2):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pipeline = FluxPipeline()
    
    def process_batch(
        self,
        prompts: List[str],
        output_dir: str,
        **generation_kwargs
    ) -> List[str]:
        """Process a batch of prompts
        
        Args:
            prompts: List of text prompts
            output_dir: Directory to save images
            **generation_kwargs: Additional generation parameters
        
        Returns:
            List of output image paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_paths = []
        
        logger.info(f"Processing {len(prompts)} prompts in batches of {self.batch_size}")
        
        for i in tqdm(range(0, len(prompts), self.batch_size), desc="Batches"):
            batch_prompts = prompts[i:i + self.batch_size]
            
            for j, prompt in enumerate(batch_prompts):
                idx = i + j
                output_path = output_dir / f"image_{idx:04d}.png"
                
                try:
                    self.pipeline(
                        prompt,
                        output_path=str(output_path),
                        **generation_kwargs
                    )
                    output_paths.append(str(output_path))
                    logger.info(f"Generated {idx + 1}/{len(prompts)}: {output_path.name}")
                
                except Exception as e:
                    logger.error(f"Failed to generate image {idx}: {e}")
                    output_paths.append(None)
        
        successful = sum(1 for p in output_paths if p is not None)
        logger.info(f"Batch processing complete: {successful}/{len(prompts)} successful")
        
        return output_paths
    
    def process_from_file(
        self,
        prompts_file: str,
        output_dir: str,
        **generation_kwargs
    ) -> List[str]:
        """Process prompts from a text file (one per line)
        
        Args:
            prompts_file: Path to file containing prompts
            output_dir: Directory to save images
            **generation_kwargs: Additional generation parameters
        
        Returns:
            List of output image paths
        """
        with open(prompts_file, 'r', encoding='utf-8') as f:
            prompts = [line.strip() for line in f if line.strip()]
        
        logger.info(f"Loaded {len(prompts)} prompts from {prompts_file}")
        
        return self.process_batch(prompts, output_dir, **generation_kwargs)
