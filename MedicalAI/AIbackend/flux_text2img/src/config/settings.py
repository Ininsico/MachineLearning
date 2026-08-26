import os

HF_TOKEN = os.getenv("HF_TOKEN", "")
MODEL_ID = "black-forest-labs/FLUX.1-schnell"

DEFAULT_HEIGHT = 1024
DEFAULT_WIDTH = 1024
DEFAULT_STEPS = 4
DEFAULT_GUIDANCE = 3.5

OUTPUT_DIR = "outputs"
CHECKPOINT_DIR = "checkpoints"
LOG_DIR = "logs"
