"""
Utility functions for logging and monitoring
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logger(name, log_file=None, level=logging.INFO):
    """Setup logger with file and console handlers"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers = []
    
    # Format
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class TrainingLogger:
    """Advanced training logger with metrics tracking"""
    
    def __init__(self, log_dir, experiment_name):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiment_name = experiment_name
        self.start_time = datetime.now()
        
        # Setup logger
        log_file = self.log_dir / f"{experiment_name}_{self.start_time.strftime('%Y%m%d_%H%M%S')}.log"
        self.logger = setup_logger(experiment_name, log_file)
        
        # Metrics storage
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'perplexity': [],
            'step': []
        }
    
    def log_step(self, step, metrics):
        """Log training step"""
        self.metrics['step'].append(step)
        
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)
        
        # Log to console
        msg = f"Step {step}"
        for key, value in metrics.items():
            if isinstance(value, float):
                msg += f" | {key}: {value:.4f}"
            else:
                msg += f" | {key}: {value}"
        
        self.logger.info(msg)
    
    def save_metrics(self):
        """Save metrics to file"""
        import json
        
        metrics_file = self.log_dir / f"{self.experiment_name}_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def get_summary(self):
        """Get training summary"""
        duration = datetime.now() - self.start_time
        
        summary = {
            'experiment': self.experiment_name,
            'duration': str(duration),
            'total_steps': len(self.metrics['step']),
            'final_train_loss': self.metrics['train_loss'][-1] if self.metrics['train_loss'] else None,
            'final_val_loss': self.metrics['val_loss'][-1] if self.metrics['val_loss'] else None,
            'best_val_loss': min(self.metrics['val_loss']) if self.metrics['val_loss'] else None
        }
        
        return summary
