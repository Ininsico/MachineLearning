import os

# File paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, 'dataset', 'student_data_clean.csv')
TEST_DATASET_PATH = os.path.join(BASE_DIR, 'dataset', 'test.csv')
RANDOM_STATE = 42
N_CLUSTERS = 3
N_ESTIMATORS = 100
MAX_DEPTH = 10
