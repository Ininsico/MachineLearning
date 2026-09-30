"""Metrics, diagnostic plots and feature-attribution analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    root_mean_squared_error,
)
from scipy.stats import spearmanr  # noqa: E402

from .logging_utils import get_logger  # noqa: E402

try:
    import seaborn as sns

    _HAS_SEABORN = True
except ImportError:
    _HAS_SEABORN = False


def classification_metrics(y_true, y_pred, y_score: Optional[np.ndarray] = None) -> Dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }
    if y_score is not None and len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
        metrics["average_precision"] = float(average_precision_score(y_true, y_score))
    else:
        metrics["roc_auc"] = float("nan")
        metrics["average_precision"] = float("nan")
    return metrics


def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    metrics = {
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }
    if len(y_true) > 2 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        metrics["spearman"] = float(spearmanr(y_true, y_pred).statistic)
    else:
        metrics["spearman"] = float("nan")
    return metrics


def _save(fig, path: Path, dpi: int = 200) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    get_logger().info("  wrote figure: %s", path.name)
    return path


def plot_learning_curve(history: pd.DataFrame, path: Path, title: str, dpi: int = 200) -> Path:
    fig, ax1 = plt.subplots(figsize=(7.5, 4.5))
    ax1.plot(history["epoch"], history["training_loss"], color="#1f77b4", lw=2, label="Training loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training loss", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.grid(alpha=0.3)

    if history["validation_score"].notna().any():
        ax2 = ax1.twinx()
        ax2.plot(history["epoch"], history["validation_score"], color="#d62728", lw=2,
                 label="Validation score")
        ax2.set_ylabel("Validation score", color="#d62728")
        ax2.tick_params(axis="y", labelcolor="#d62728")
        lines = ax1.get_lines() + ax2.get_lines()
        ax1.legend(lines, [line.get_label() for line in lines], loc="center right", fontsize=9)
    else:
        ax1.legend(loc="center right", fontsize=9)

    ax1.set_title(title)
    return _save(fig, path, dpi)


def plot_confusion_matrix(cm: np.ndarray, labels: Sequence[str], path: Path,
                          title: str, dpi: int = 200) -> Path:
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    if _HAS_SEABORN:
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=labels, yticklabels=labels, ax=ax)
    else:
        im = ax.imshow(cm, cmap="Blues")
        fig.colorbar(im, ax=ax)
        ax.set_xticks(range(len(labels)), labels)
        ax.set_yticks(range(len(labels)), labels)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    return _save(fig, path, dpi)


def plot_roc_curve(y_true, y_score, auc_value: float, path: Path, title: str, dpi: int = 200) -> Path:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    ax.plot(fpr, tpr, color="#1f77b4", lw=2, label=f"Model (AUC = {auc_value:.3f})")
    ax.plot([0, 1], [0, 1], color="grey", ls="--", lw=1, label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    return _save(fig, path, dpi)


def plot_precision_recall_curve(y_true, y_score, ap_value: float, path: Path,
                                title: str, dpi: int = 200) -> Path:
    from sklearn.metrics import precision_recall_curve

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    prevalence = float(np.mean(y_true))
    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    ax.plot(recall, precision, color="#9467bd", lw=2, label=f"Model (AP = {ap_value:.3f})")
    ax.axhline(prevalence, color="grey", ls="--", lw=1,
               label=f"No-skill (positive rate = {prevalence:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    return _save(fig, path, dpi)


def plot_ranked_importance(frame: pd.DataFrame, value_column: str, path: Path,
                           title: str, top_n: int = 25, dpi: int = 200,
                           error_column: Optional[str] = None) -> Path:
    subset = frame.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, max(3.0, 0.28 * len(subset) + 1.2)))
    errors = subset[error_column].to_numpy() if error_column and error_column in subset else None
    ax.barh(subset["feature"].astype(str), subset[value_column].to_numpy(),
            xerr=errors, color="#2ca02c", alpha=0.85)
    ax.set_xlabel(value_column.replace("_", " "))
    ax.set_title(title)
    ax.grid(alpha=0.3, axis="x")
    return _save(fig, path, dpi)


def permutation_importance_table(
    pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    cfg,
    seed: int,
    scoring: str = "roc_auc",
) -> pd.DataFrame:
    """Model-agnostic permutation importance on the supplied (test) data."""
    ecfg = cfg.evaluation.permutation_importance
    with np.errstate(all="ignore"):
        result = permutation_importance(
            pipeline, X.to_numpy(), y.to_numpy(),
            scoring=scoring, n_repeats=int(ecfg.n_repeats),
            random_state=seed, n_jobs=1,
        )
    frame = pd.DataFrame(
        {
            "feature": list(X.columns),
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False).reset_index(drop=True)
    frame["rank"] = np.arange(1, len(frame) + 1)
    return frame


def shap_importance_table(pipeline, X: pd.DataFrame, cfg, seed: int) -> Optional[pd.DataFrame]:
    """Mean |SHAP| per feature for the fitted model, if the shap package is usable.

    The explainer is given a callable that runs the *whole* pipeline (scaling and
    in-fold feature selection included), so attributions are expressed in the
    original feature space rather than in the selected sub-space.
    """
    try:
        import shap
    except ImportError:
        get_logger().warning("shap not installed - skipping SHAP attribution")
        return None

    scfg = cfg.evaluation.shap
    rng = np.random.default_rng(seed)
    values = X.to_numpy()
    if values.shape[0] > int(scfg.max_samples):
        index = rng.choice(values.shape[0], size=int(scfg.max_samples), replace=False)
        values = values[index]

    def predict_positive(data: np.ndarray) -> np.ndarray:
        return pipeline.predict_proba(np.asarray(data))[:, 1]

    try:
        background = values[rng.choice(values.shape[0], size=min(50, values.shape[0]), replace=False)]
        explainer = shap.PermutationExplainer(predict_positive, background, seed=seed)
        attributions = explainer(values, max_evals=max(2 * X.shape[1] + 1, 400), silent=True).values
        attributions = np.asarray(attributions)
        if attributions.ndim == 3:
            attributions = attributions[:, :, -1]
    except Exception as exc:  # pragma: no cover - depends on shap version
        get_logger().warning("SHAP attribution failed (%s: %s) - falling back to permutation importance",
                             type(exc).__name__, str(exc)[:160])
        return None

    frame = pd.DataFrame(
        {"feature": list(X.columns), "mean_abs_shap": np.abs(attributions).mean(axis=0)}
    ).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    frame["rank"] = np.arange(1, len(frame) + 1)
    return frame


def summarise_confusion(cm: np.ndarray) -> Dict[str, float]:
    """Derive rates from a 2x2 confusion matrix.

    Balanced accuracy is the mean of sensitivity and specificity. Compute the two
    terms explicitly: writing the sum inside a conditional expression silently
    mis-binds, because ``+`` binds tighter than the ternary, halving the result.
    """
    tn, fp, fn, tp = (cm.ravel() if cm.size == 4 else (0, 0, 0, 0))
    tn, fp, fn, tp = float(tn), float(fp), float(fn), float(tp)

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")

    if np.isnan(sensitivity) or np.isnan(specificity):
        balanced = float("nan")
    else:
        balanced = (sensitivity + specificity) / 2.0

    return {
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": balanced,
        "n": float(cm.sum()),
    }


def confusion_matrix_frame(y_true, y_pred, labels: Sequence[str] = ("0", "1")) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return pd.DataFrame(cm, index=[f"true_{l}" for l in labels], columns=[f"pred_{l}" for l in labels])


def tune_threshold(y_true, y_score, metric: str = "f1") -> Tuple[float, float]:
    """Pick the decision threshold that maximises a metric on the given scores.

    The positive class here is the minority class (documented metal-active genera
    are a small share of any community), so the default 0.5 threshold makes the
    model predict almost everything negative and F1 collapses to ~0. Tuning the
    threshold is standard practice for imbalanced problems; it is fitted on
    out-of-fold training predictions and then applied unchanged to the test set,
    so the reported test metrics remain honest.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if len(np.unique(y_true)) < 2:
        return 0.5, float("nan")

    candidates = np.unique(np.round(y_score, 6))
    if len(candidates) > 500:
        candidates = np.quantile(y_score, np.linspace(0.01, 0.99, 500))

    best_threshold, best_value = 0.5, -np.inf
    for threshold in candidates:
        predicted = (y_score >= threshold).astype(int)
        if metric == "f1":
            value = f1_score(y_true, predicted, zero_division=0)
        elif metric == "balanced_accuracy":
            value = balanced_accuracy_score(y_true, predicted)
        elif metric == "mcc":
            value = matthews_corrcoef(y_true, predicted)
        else:
            raise ValueError(f"Unsupported threshold metric: {metric}")
        if value > best_value:
            best_value, best_threshold = float(value), float(threshold)
    return best_threshold, best_value


def out_of_fold_scores(pipeline, X: pd.DataFrame, y: pd.Series, cfg, task: str = "classification") -> np.ndarray:
    """Cross-validated probabilities used to fit the decision threshold.

    Scores come from ``cross_val_predict``, so every score is produced by a model
    that never saw that row, and the resulting threshold is not fitted on the test
    split.
    """
    from sklearn.model_selection import cross_val_predict

    from .models import make_cv

    return cross_val_predict(
        pipeline, X.to_numpy(), y.to_numpy(), cv=make_cv(cfg),
        method="predict_proba", n_jobs=1,
    )[:, 1]


def majority_class_reference(y: pd.Series) -> Dict[str, float]:
    """Metrics obtained by always predicting the majority class."""
    y = pd.Series(y).to_numpy()
    majority = int(pd.Series(y).mode().iloc[0])
    predicted = np.full_like(y, majority)
    return {
        "majority_class": majority,
        "accuracy": float(accuracy_score(y, predicted)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "positive_rate": float((y == 1).mean()),
    }
