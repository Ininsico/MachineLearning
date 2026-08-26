import os

BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(BASE_DIR, "data")
RAW_DATA_PATH   = os.path.join(BASE_DIR, "02-14-2018.csv")
PROCESSED_DIR   = os.path.join(DATA_DIR, "processed")
RESULTS_DIR     = os.path.join(BASE_DIR, "results")
PLOTS_DIR       = os.path.join(RESULTS_DIR, "plots")
REPORTS_DIR     = os.path.join(RESULTS_DIR, "reports")
MODELS_DIR      = os.path.join(RESULTS_DIR, "models")
LOGS_DIR        = os.path.join(BASE_DIR, "logs")

LABEL_COL       = "Label"
SAMPLE_SIZE     = 200_000
RANDOM_STATE    = 42
TEST_SIZE       = 0.20
VAL_SIZE        = 0.10

LABEL_MAP = {
    "Benign":           0,
    "FTP-BruteForce":   1,
    "SSH-Bruteforce":   1,
}

CLASS_NAMES     = ["Benign", "Anomaly"]
ATTACK_TYPES    = ["FTP-BruteForce", "SSH-Bruteforce"]

LATENT_DIM      = 64
GAN_EPOCHS      = 200
GAN_BATCH_SIZE  = 256
GAN_LR_G        = 2e-4
GAN_LR_D        = 2e-4
GAN_BETAS       = (0.5, 0.999)
GAN_SYNTHETIC_SAMPLES = 5_000

CLF_EPOCHS      = 30
CLF_BATCH_SIZE  = 512
CLF_LR          = 1e-3
CLF_HIDDEN      = [256, 128, 64]
CLF_DROPOUT     = 0.3
CLF_WEIGHT_DECAY = 1e-4

BERT_MODEL      = "bert-base-uncased"
BERT_MAX_LEN    = 128
BERT_BATCH_SIZE = 16
BERT_EPOCHS     = 3
BERT_LR         = 2e-5
NUM_EXPLAIN_SAMPLES = 10

METRICS         = ["accuracy", "precision", "recall", "f1"]

FIGURE_DPI      = 150
PALETTE         = {
    "primary":      "#6C63FF",
    "secondary":    "#FF6584",
    "success":      "#43AA8B",
    "warning":      "#F9C74F",
    "danger":       "#F94144",
    "background":   "#0F0F1A",
    "surface":      "#1A1A2E",
    "text":         "#E0E0FF",
}