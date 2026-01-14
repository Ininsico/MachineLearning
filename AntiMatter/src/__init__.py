"""
Initialize the src package
"""

__version__ = "1.0.0"
__author__ = "ML Research Team"

from pathlib import Path

# Package root
PACKAGE_ROOT = Path(__file__).parent.parent

# Data directories
DATA_DIR = PACKAGE_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
VOCAB_DIR = DATA_DIR / "vocab"

# Model directories
CHECKPOINT_DIR = PACKAGE_ROOT / "checkpoints"
RESULTS_DIR = PACKAGE_ROOT / "results"

# Logs
LOGS_DIR = PACKAGE_ROOT / "logs"

# Create directories if they don't exist
for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, VOCAB_DIR, 
                 CHECKPOINT_DIR, RESULTS_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)
