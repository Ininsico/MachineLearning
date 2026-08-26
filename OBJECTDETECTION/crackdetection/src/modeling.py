import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score

def train_logistic_regression(X_scaled, y):
    model = LogisticRegression(random_state=42)
    model.fit(X_scaled, y)
    return model

def calculate_unscaled_coefficients(model, scaler):
    weights = model.coef_[0]
    intercept = model.intercept_[0]
    means = scaler.mean_
    scales = scaler.scale_
    
    unscaled_weights = weights / scales
    unscaled_intercept = intercept - np.sum((weights * means) / scales)
    
    return unscaled_intercept, unscaled_weights

def analyze_thresholds(model, X_scaled, y):
    y_probs = model.predict_proba(X_scaled)[:, 1]
    thresholds = np.arange(0.1, 1.0, 0.1)
    results = []
    
    for t in thresholds:
        y_pred = (y_probs >= t).astype(int)
        results.append({
            'Threshold': round(t, 1),
            'Accuracy': accuracy_score(y, y_pred),
            'Precision': precision_score(y, y_pred, zero_division=0),
            'Recall': recall_score(y, y_pred, zero_division=0),
            'Crack_Yes': np.sum(y_pred),
            'Crack_No': len(y_pred) - np.sum(y_pred)
        })
    
    results_df = pd.DataFrame(results)
    results_df['Ratio_Y2N'] = results_df['Crack_Yes'] / results_df['Crack_No']
    return results_df, y_probs
