from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "neo.csv"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

RANDOM_SEED = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.1

NUM_EPOCHS = 40
BATCH_SIZE = 1024
LEARNING_RATE = 1e-3
HIDDEN_DIMS = [128, 64, 32]
DROPOUT = 0.3

# Column definitions
ID_COLS = ["id", "name"]
TARGET_COL = "hazardous"
NUMERIC_COLS = [
    "est_diameter_min",
    "est_diameter_max",
    "relative_velocity",
    "miss_distance",
    "absolute_magnitude",
]
CAT_COLS = ["orbiting_body", "sentry_object"]
ENGINEERED_COLS = ["est_diameter_mean", "est_diameter_span"]

DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"
