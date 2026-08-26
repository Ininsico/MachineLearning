from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import pandas as pd

def train_model(X, y):
    model = LogisticRegression()
    model.fit(X, y)
    return model

def analyze_thresholds(model, X, y, thresholds):
    probs = model.predict_proba(X)[:, 1]
    results = []
    
    for t in thresholds:
        preds = (probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, preds).ravel()
        
        results.append({
            'Threshold': t,
            'Accuracy': accuracy_score(y, preds),
            'Precision': precision_score(y, preds, zero_division=0),
            'Recall': recall_score(y, preds, zero_division=0),
            'F1': f1_score(y, preds, zero_division=0),
            'FP': fp,
            'FN': fn,
            'TN': tn,
            'TP': tp
        })
    return pd.DataFrame(results)

def manual_sigmoid(b0, coefficients, features):
    z = b0 + np.dot(coefficients, features)
    return 1 / (1 + np.exp(-z))
