"""
Training script for Custom 300M Language Model
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import yaml
import os
import time
from tqdm import tqdm
import wandb
from pathlib import Path

from model.architecture import CustomLM
from preprocessing.dataset import TextDataset
from utils.logger import setup_logger
from utils.metrics import calculate_perplexity


class Trainer:
    def __init__(self, config_path):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup logger
        self.logger = setup_logger('training')
        self.logger.info("Initializing training...")
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")
        
        # Initialize model
        self.model = CustomLM(self.config['model']).to(self.device)
        self.logger.info(f"Model initialized with {self.model.get_num_params():,} parameters")
        
        # Multi-GPU setup
        if torch.cuda.device_count() > 1:
            self.logger.info(f"Using {torch.cuda.device_count()} GPUs")
            self.model = nn.DataParallel(self.model)
        
        # Setup optimizer
        self.optimizer = self._configure_optimizer()
        
        # Setup learning rate scheduler
        self.scheduler = self._configure_scheduler()
        
        # Mixed precision training
        self.scaler = GradScaler() if self.config['training']['mixed_precision'] else None
        
        # Load datasets
        self.train_loader, self.val_loader = self._load_datasets()
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        
        # Create checkpoint directory
        os.makedirs(self.config['training']['checkpoint_dir'], exist_ok=True)
        
    def _configure_optimizer(self):
        """Configure AdamW optimizer with weight decay"""
        # Separate parameters for weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'bias' in name or 'ln' in name or 'LayerNorm' in name:
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)
        
        optimizer_grouped_parameters = [
            {'params': decay_params, 'weight_decay': self.config['training']['weight_decay']},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]
        
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.config['training']['learning_rate'],
            betas=(self.config['training']['beta1'], self.config['training']['beta2']),
            eps=self.config['training']['epsilon']
        )
        
        return optimizer
    
    def _configure_scheduler(self):
        """Configure cosine learning rate scheduler with warmup"""
        from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
        
        warmup_steps = self.config['training']['warmup_steps']
        total_steps = self.config['training']['total_steps']
        
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
        scheduler = LambdaLR(self.optimizer, lr_lambda)
        return scheduler
    
    def _load_datasets(self):
        """Load training and validation datasets"""
        train_dataset = TextDataset(
            self.config['data']['train_data_path'],
            max_length=self.config['data']['max_seq_length']
        )
        
        val_dataset = TextDataset(
            self.config['data']['val_data_path'],
            max_length=self.config['data']['max_seq_length']
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=self.config['data']['num_workers'],
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=self.config['data']['num_workers'],
            pin_memory=True
        )
        
        return train_loader, val_loader
    
    def train_step(self, batch):
        """Single training step"""
        input_ids = batch['input_ids'].to(self.device)
        labels = batch['labels'].to(self.device)
        
        if self.scaler:
            with autocast():
                _, loss = self.model(input_ids, labels=labels)
        else:
            _, loss = self.model(input_ids, labels=labels)
        
        # Backward pass
        if self.scaler:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config['training']['max_grad_norm']
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config['training']['max_grad_norm']
            )
            self.optimizer.step()
        
        self.optimizer.zero_grad()
        self.scheduler.step()
        
        return loss.item()
    
    def validate(self):
        """Validation loop"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                _, loss = self.model(input_ids, labels=labels)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        perplexity = calculate_perplexity(avg_loss)
        
        self.model.train()
        return avg_loss, perplexity
    
    def save_checkpoint(self, step, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'step': step,
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config['model'],
            'best_val_loss': self.best_val_loss
        }
        
        checkpoint_path = os.path.join(
            self.config['training']['checkpoint_dir'],
            f'checkpoint_step_{step}.pt'
        )
        
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        if is_best:
            best_path = os.path.join(
                self.config['training']['checkpoint_dir'],
                'best_model.pt'
            )
            torch.save(checkpoint, best_path)
            self.logger.info(f"Best model saved: {best_path}")
    
    def train(self):
        """Main training loop"""
        self.logger.info("Starting training...")
        self.model.train()
        
        total_steps = self.config['training']['total_steps']
        logging_steps = self.config['training']['logging_steps']
        eval_steps = self.config['training']['eval_steps']
        save_steps = self.config['training']['save_steps']
        
        running_loss = 0
        start_time = time.time()
        
        while self.global_step < total_steps:
            for batch in self.train_loader:
                loss = self.train_step(batch)
                running_loss += loss
                self.global_step += 1
                
                # Logging
                if self.global_step % logging_steps == 0:
                    avg_loss = running_loss / logging_steps
                    elapsed = time.time() - start_time
                    lr = self.scheduler.get_last_lr()[0]
                    
                    self.logger.info(
                        f"Step {self.global_step}/{total_steps} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"LR: {lr:.2e} | "
                        f"Time: {elapsed:.2f}s"
                    )
                    
                    running_loss = 0
                    start_time = time.time()
                
                # Validation
                if self.global_step % eval_steps == 0:
                    val_loss, perplexity = self.validate()
                    self.logger.info(
                        f"Validation | Loss: {val_loss:.4f} | Perplexity: {perplexity:.2f}"
                    )
                    
                    # Save best model
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.save_checkpoint(self.global_step, is_best=True)
                
                # Save checkpoint
                if self.global_step % save_steps == 0:
                    self.save_checkpoint(self.global_step)
                
                if self.global_step >= total_steps:
                    break
            
            self.epoch += 1
        
        self.logger.info("Training complete!")
        self.save_checkpoint(self.global_step)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/model_config.yaml')
    args = parser.parse_args()
    
    trainer = Trainer(args.config)
    trainer.train()
