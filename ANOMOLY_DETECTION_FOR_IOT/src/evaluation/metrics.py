import logging
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report,
)

logger = logging.getLogger(__name__)

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_proba: np.ndarray = None, prefix: str = "") -> dict:

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="binary", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="binary", zero_division=0)
    f1   = f1_score(y_true, y_pred, average="binary", zero_division=0)

    metrics = {
        f"{prefix}accuracy":  acc,
        f"{prefix}precision": prec,
        f"{prefix}recall":    rec,
        f"{prefix}f1_score":  f1,
    }

    if y_proba is not None:
        try:
            metrics[f"{prefix}roc_auc"] = roc_auc_score(y_true, y_proba)
            metrics[f"{prefix}pr_auc"]  = average_precision_score(y_true, y_proba)
        except Exception:
            pass

    prec_pc  = precision_score(y_true, y_pred, average=None, zero_division=0)
    rec_pc   = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_pc    = f1_score(y_true, y_pred, average=None, zero_division=0)
    for i, cls in enumerate(["Benign", "Anomaly"]):
        metrics[f"{prefix}{cls}_precision"] = prec_pc[i] if i < len(prec_pc) else 0.0
        metrics[f"{prefix}{cls}_recall"]    = rec_pc[i]  if i < len(rec_pc)  else 0.0
        metrics[f"{prefix}{cls}_f1"]        = f1_pc[i]   if i < len(f1_pc)   else 0.0

    logger.info(" %sMetrics:", prefix or "")
    for k, v in metrics.items():
        logger.info("   %-35s %.4f", k, v)

    return metrics

def get_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return confusion_matrix(y_true, y_pred)

def get_classification_report(y_true: np.ndarray, y_pred: np.ndarray,
                               class_names: list = None) -> str:
    target_names = class_names or ["Benign", "Anomaly"]
    return classification_report(y_true, y_pred, target_names=target_names, zero_division=0)

def build_comparison_table(results_dict: dict) -> pd.DataFrame:

    rows = []
    for model_name, metrics in results_dict.items():
        row = {"Model": model_name}
        for col in ["accuracy", "precision", "recall", "f1_score", "roc_auc", "pr_auc"]:
            row[col.replace("_", " ").title()] = f"{metrics.get(col, 0.0):.4f}"
        rows.append(row)

    df = pd.DataFrame(rows).set_index("Model")
    logger.info("\n Model Comparison Table:\n%s", df.to_string())
    return df