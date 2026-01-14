"""
Checkpoint management utilities
"""

import torch
import os
import shutil
from pathlib import Path
from typing import Optional, Dict
import json


class CheckpointManager:
    """Manage model checkpoints during training"""
    
    def __init__(self, checkpoint_dir, max_checkpoints=5):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        
        self.checkpoints = []
        self.best_checkpoint = None
        self.best_metric = float('inf')
    
    def save_checkpoint(self, 
                       model, 
                       optimizer, 
                       scheduler,
                       step,
                       epoch,
                       metrics,
                       config,
                       is_best=False):
        """Save a training checkpoint"""
        
        checkpoint = {
            'step': step,
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'metrics': metrics,
            'config': config
        }
        
        # Save checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_step_{step}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Track checkpoint
        self.checkpoints.append({
            'path': checkpoint_path,
            'step': step,
            'metric': metrics.get('val_loss', float('inf'))
        })
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            shutil.copy(checkpoint_path, best_path)
            self.best_checkpoint = checkpoint_path
            self.best_metric = metrics.get('val_loss', float('inf'))
        
        # Clean up old checkpoints
        self._cleanup_checkpoints()
        
        return checkpoint_path
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints keeping only the most recent ones"""
        if len(self.checkpoints) > self.max_checkpoints:
            # Sort by step
            self.checkpoints.sort(key=lambda x: x['step'])
            
            # Remove oldest checkpoints
            while len(self.checkpoints) > self.max_checkpoints:
                old_checkpoint = self.checkpoints.pop(0)
                
                # Don't delete best checkpoint
                if old_checkpoint['path'] != self.best_checkpoint:
                    if old_checkpoint['path'].exists():
                        old_checkpoint['path'].unlink()
    
    def load_checkpoint(self, checkpoint_path, model, optimizer=None, scheduler=None):
        """Load a checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        return checkpoint
    
    def get_latest_checkpoint(self):
        """Get path to latest checkpoint"""
        if not self.checkpoints:
            return None
        
        latest = max(self.checkpoints, key=lambda x: x['step'])
        return latest['path']
    
    def get_best_checkpoint(self):
        """Get path to best checkpoint"""
        best_path = self.checkpoint_dir / 'best_model.pt'
        return best_path if best_path.exists() else None
    
    def list_checkpoints(self):
        """List all available checkpoints"""
        return sorted(self.checkpoints, key=lambda x: x['step'])
    
    def save_metadata(self):
        """Save checkpoint metadata"""
        metadata = {
            'checkpoints': [
                {
                    'path': str(cp['path']),
                    'step': cp['step'],
                    'metric': cp['metric']
                }
                for cp in self.checkpoints
            ],
            'best_checkpoint': str(self.best_checkpoint) if self.best_checkpoint else None,
            'best_metric': self.best_metric
        }
        
        metadata_path = self.checkpoint_dir / 'checkpoints_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)


def convert_checkpoint_to_huggingface(checkpoint_path, output_dir):
    """Convert custom checkpoint to HuggingFace format"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save model weights
    model_path = output_path / 'pytorch_model.bin'
    torch.save(checkpoint['model_state_dict'], model_path)
    
    # Save config
    config = checkpoint['config']
    config_path = output_path / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Checkpoint converted to HuggingFace format: {output_dir}")


if __name__ == "__main__":
    print("Checkpoint manager loaded")
