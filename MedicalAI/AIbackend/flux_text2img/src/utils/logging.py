import logging
import sys
from pathlib import Path
from typing import Optional

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Get configured logger
    
    Args:
        name: Logger name (usually __name__)
        level: Logging level
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        logger.setLevel(level)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger

def setup_file_logging(log_dir: str, name: str = "flux_t2i") -> logging.Logger:
    """Setup file logging
    
    Args:
        log_dir: Directory for log files
        name: Logger name
    
    Returns:
        Logger with file handler
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logger = get_logger(name)
    
    file_handler = logging.FileHandler(log_dir / f"{name}.log")
    file_handler.setLevel(logging.DEBUG)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    
    return logger
