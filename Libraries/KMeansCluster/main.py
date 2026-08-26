import os
os.environ["LOKY_MAX_CPU_COUNT"] = "1" # Silence joblib warning

from src.data_processor import DataProcessor
from src.models import PerformancePredictor
from src.reporter import Reporter
from src.config import DATASET_PATH, TEST_DATASET_PATH

def run_pipeline():
    processor = DataProcessor()
    predictor = PerformancePredictor()
    
    # 1. Train on main dataset
    raw_df = processor.clean_raw_data(DATASET_PATH)
    sem_df = processor.aggregate_to_semesters(raw_df)
    X_train, y_train = predictor.prepare_training_data(sem_df)
    predictor.train(X_train, y_train)
    
    # Internal Performance
    train_metrics = predictor.evaluate(X_train, y_train)
    Reporter.display_model_evaluation(train_metrics, "INTERNAL TRAIN STATS")

    # 2. External Test (Student 7)
    if os.path.exists(TEST_DATASET_PATH):
        test_df = processor.clean_raw_data(TEST_DATASET_PATH)
        test_sem_df = processor.aggregate_to_semesters(test_df)
        X_test, y_test = predictor.prepare_training_data(test_sem_df)
        
        test_metrics = predictor.evaluate(X_test, y_test)
        Reporter.display_model_evaluation(test_metrics, "EXTERNAL TEST (UNSEEN)")
    else:
        print(f"Skipping test: {TEST_DATASET_PATH} not found.")

if __name__ == "__main__":
    run_pipeline()