import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from src.config import RANDOM_STATE, N_ESTIMATORS, N_CLUSTERS

class PerformancePredictor:
    def __init__(self):
        self.model = Ridge(alpha=5.0)
        self.scaler = StandardScaler()
    
    def prepare_training_data(self, sem_df):
        features, targets = [], []
        for student in sem_df['student_id'].unique():
            history = sem_df[sem_df['student_id'] == student]
            for i in range(len(history) - 1):
                curr = history.iloc[i]
                nxt = history.iloc[i+1]
                # Added Cumulative GPA as a stabilizing feature
                features.append([
                    curr['avg_marks'], 
                    curr['gpa'], 
                    curr['cum_gpa'],
                    curr['gpa_change'],
                    curr['prog_ratio'],
                    curr['total_credits'], 
                    curr['retakes'], 
                    nxt['semester']
                ])
                targets.append(nxt['avg_marks'])
        return np.array(features), np.array(targets)

    def train(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

    def evaluate(self, X, y):
        from sklearn.metrics import mean_absolute_error, r2_score
        X_scaled = self.scaler.transform(X)
        y_pred = self.model.predict(X_scaled)
        return {
            "r2": r2_score(y, y_pred),
            "mae": mean_absolute_error(y, y_pred),
            "accuracy": np.mean(np.abs((y - y_pred) / y) < 0.1) * 100
        }
