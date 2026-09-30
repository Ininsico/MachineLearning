"""Model construction, hyper-parameter search and cross-validation.

Architecture choice: a feed-forward multilayer perceptron (MLP). The inputs are
static community snapshots - one fixed-length vector of taxon abundances per
observation - so sequence models (LSTM/GRU) and image models (CNN) are
inapplicable by construction, and transformer architectures would add capacity
without any structural justification. This is stated explicitly in the README.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline

from .features import build_selector, make_scaler
from .logging_utils import get_logger


def _as_hidden_layers(value) -> tuple:
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return ()
        if all(isinstance(v, (list, tuple)) for v in value):
            return tuple(int(v) for v in value[0])
        return tuple(int(v) for v in value)
    return (int(value),)


def resolve_backend(cfg) -> str:
    """Return the MLP backend actually in use: 'sklearn' or 'torch'.

    If torch is requested but not importable, this falls back to the scikit-learn MLP
    with a warning rather than failing the run, so a configuration file can be shared
    between a GPU machine and one without PyTorch.
    """
    backend = str(cfg.model.get("backend", "sklearn") or "sklearn").strip().lower()
    if backend not in {"sklearn", "torch"}:
        raise ValueError(f"Unknown model.backend: {backend!r} (expected 'sklearn' or 'torch')")

    if backend == "torch":
        from .torch_backend import TORCH_AVAILABLE

        if not TORCH_AVAILABLE:
            get_logger().warning(
                "model.backend is 'torch' but PyTorch is not installed; falling back to the "
                "scikit-learn MLPClassifier. Install torch to use the GPU backend."
            )
            return "sklearn"
    return backend


def effective_grid_n_jobs(cfg) -> int:
    """Avoid fanning a GridSearchCV across CPU workers when a single GPU is shared.

    With the torch backend on CUDA, ``n_jobs=-1`` would launch several processes
    contending for one device while each model is far too small to saturate it, so
    the search is forced serial.
    """
    n_jobs = int(cfg.model.grid.n_jobs)
    if resolve_backend(cfg) == "torch":
        from .torch_backend import resolve_device

        if resolve_device(str(getattr(cfg.model.torch, "device", "auto"))) == "cuda":
            if n_jobs != 1:
                get_logger().info(
                    "torch backend on CUDA: forcing GridSearchCV n_jobs=1 (was %d) to avoid "
                    "multiple processes contending for one GPU", n_jobs,
                )
            return 1
    return n_jobs


def make_mlp_classifier(cfg, seed: int, **overrides) -> BaseEstimator:
    """Construct the classifier, honouring the configured backend."""
    mcfg = cfg.model.mlp_classifier
    hidden = _as_hidden_layers(mcfg.hidden_layer_sizes)

    if resolve_backend(cfg) == "torch":
        from .torch_backend import TorchMLPClassifier, describe_device, resolve_device

        tcfg = getattr(cfg.model, "torch", None)
        device_preference = str(getattr(tcfg, "device", "auto")) if tcfg else "auto"
        resolved = resolve_device(device_preference)
        get_logger().info("MLP backend: torch on %s", describe_device(resolved))
        params = dict(
            hidden_layer_sizes=hidden,
            activation=str(mcfg.activation),
            alpha=float(mcfg.alpha),
            learning_rate_init=float(mcfg.learning_rate_init),
            batch_size=int(mcfg.batch_size),
            max_iter=int(mcfg.max_iter),
            early_stopping=bool(mcfg.early_stopping),
            validation_fraction=float(mcfg.validation_fraction),
            n_iter_no_change=int(mcfg.n_iter_no_change),
            tol=float(mcfg.tol),
            dropout=float(getattr(tcfg, "dropout", 0.0)) if tcfg else 0.0,
            class_weight=getattr(tcfg, "class_weight", None) if tcfg else None,
            device=device_preference,
            deterministic=bool(getattr(tcfg, "deterministic", True)) if tcfg else True,
            random_state=seed,
        )
        params.update(overrides)
        return TorchMLPClassifier(**params)

    params = dict(
        hidden_layer_sizes=hidden,
        activation=str(mcfg.activation),
        solver=str(mcfg.solver),
        alpha=float(mcfg.alpha),
        learning_rate=str(mcfg.learning_rate),
        learning_rate_init=float(mcfg.learning_rate_init),
        early_stopping=bool(mcfg.early_stopping),
        validation_fraction=float(mcfg.validation_fraction),
        n_iter_no_change=int(mcfg.n_iter_no_change),
        batch_size=int(mcfg.batch_size),
        max_iter=int(mcfg.max_iter),
        tol=float(mcfg.tol),
        random_state=seed,
    )
    params.update(overrides)
    return MLPClassifier(**params)


def make_mlp_regressor(cfg, seed: int, **overrides) -> MLPRegressor:
    mcfg = cfg.model.mlp_regressor
    params = dict(
        hidden_layer_sizes=_as_hidden_layers(mcfg.hidden_layer_sizes),
        activation=str(mcfg.activation),
        solver=str(mcfg.solver),
        alpha=float(mcfg.alpha),
        learning_rate=str(mcfg.learning_rate),
        learning_rate_init=float(mcfg.learning_rate_init),
        early_stopping=bool(mcfg.early_stopping),
        validation_fraction=float(mcfg.validation_fraction),
        n_iter_no_change=int(mcfg.n_iter_no_change),
        batch_size=int(mcfg.batch_size),
        max_iter=int(mcfg.max_iter),
        random_state=seed,
    )
    params.update(overrides)
    return MLPRegressor(**params)


def make_baselines(cfg, seed: int) -> Dict[str, BaseEstimator]:
    """Reference models the MLP must beat to justify its complexity."""
    models: Dict[str, BaseEstimator] = {}
    requested = [str(name) for name in (cfg.model.baselines or [])]

    if "random_forest" in requested:
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        rcfg = cfg.model.random_forest
        models["random_forest_classifier"] = RandomForestClassifier(
            n_estimators=int(rcfg.n_estimators),
            max_depth=None if rcfg.max_depth in (None, "null") else int(rcfg.max_depth),
            min_samples_leaf=int(rcfg.min_samples_leaf),
            random_state=seed,
            n_jobs=-1,
        )
        models["random_forest_regressor"] = RandomForestRegressor(
            n_estimators=int(rcfg.n_estimators),
            max_depth=None if rcfg.max_depth in (None, "null") else int(rcfg.max_depth),
            min_samples_leaf=int(rcfg.min_samples_leaf),
            random_state=seed,
            n_jobs=-1,
        )

    if "xgboost" in requested:
        try:
            from xgboost import XGBClassifier, XGBRegressor

            xcfg = cfg.model.xgboost
            common = dict(
                n_estimators=int(xcfg.n_estimators),
                max_depth=int(xcfg.max_depth),
                learning_rate=float(xcfg.learning_rate),
                subsample=float(xcfg.subsample),
                colsample_bytree=float(xcfg.colsample_bytree),
                random_state=seed,
                n_jobs=-1,
            )
            models["xgboost_classifier"] = XGBClassifier(eval_metric="logloss", **common)
            models["xgboost_regressor"] = XGBRegressor(**common)
        except ImportError:
            get_logger().warning("xgboost not installed - baseline omitted")

    if "logistic_regression" in requested:
        lcfg = cfg.model.logistic_regression
        models["logistic_regression"] = LogisticRegression(
            C=float(lcfg.C), max_iter=int(lcfg.max_iter), random_state=seed, n_jobs=-1
        )

    return models


def build_pipeline(estimator: BaseEstimator, cfg, task: str, n_features: int) -> Pipeline:
    """Wrap an estimator with scaling and (optionally) in-fold feature selection."""
    steps: List[Tuple[str, object]] = []
    scaler = make_scaler(cfg)
    if scaler is not None:
        steps.append(("scaler", scaler))
    selector = build_selector(cfg, task, n_features)
    if selector is not None:
        steps.append(("select", selector))
    steps.append(("model", estimator))
    return Pipeline(steps)


def make_cv(cfg, n_splits: Optional[int] = None) -> StratifiedKFold:
    ccfg = cfg.model.cv
    return StratifiedKFold(
        n_splits=int(n_splits or ccfg.folds),
        shuffle=bool(ccfg.shuffle),
        random_state=int(cfg.project.seed),
    )


CLASSIFICATION_SCORERS = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
    "average_precision": "average_precision",
}

REGRESSION_SCORERS = {
    "r2": "r2",
    "neg_rmse": "neg_root_mean_squared_error",
    "neg_mae": "neg_mean_absolute_error",
}


@dataclass
class ModelResult:
    name: str
    cv_summary: pd.DataFrame
    fold_scores: pd.DataFrame
    fitted_pipeline: Optional[object] = None
    fit_seconds: float = 0.0
    best_params: Optional[dict] = None
    extra: dict = field(default_factory=dict)


def _log_cv_summary(name: str, summary: pd.DataFrame) -> None:
    logger = get_logger()
    parts = ", ".join(f"{row.metric}={row.mean:.4f}+/-{row.std:.4f}" for row in summary.itertuples())
    logger.info("  %-28s %s", name, parts)


def assess_class_balance(y: pd.Series, requested_folds: int) -> Tuple[int, int, str]:
    """Decide how many CV folds the label balance can actually support.

    Stratified k-fold needs at least ``k`` members of the minority class, so the
    requested fold count is capped by the minority count. Returning 0 means the
    minority class is too small for any stratified cross-validation, which the
    caller must report rather than work around.

    Returns ``(minority_count, usable_folds, message)``.
    """
    counts = pd.Series(y).value_counts()
    if counts.empty or len(counts) < 2:
        return 0, 0, "only one class present in the label vector"

    minority = int(counts.min())
    if minority < 2:
        return minority, 0, (
            f"the minority class has only {minority} member(s); stratified cross-validation "
            "and a stratified train/test split are both impossible"
        )
    folds = max(2, min(int(requested_folds), minority))
    message = f"minority class has {minority} members -> using {folds}-fold stratified CV"
    if folds < int(requested_folds):
        message += (
            f" (capped down from the requested {int(requested_folds)}; every reported CV metric "
            "therefore rests on very few positive examples)"
        )
    return minority, folds, message


def cross_validate_model(
    name: str,
    estimator: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    cfg,
    task: str = "classification",
    n_splits: Optional[int] = None,
) -> ModelResult:
    logger = get_logger()
    pipeline = build_pipeline(estimator, cfg, task, X.shape[1])
    cv = make_cv(cfg, n_splits)
    scoring = CLASSIFICATION_SCORERS if task == "classification" else REGRESSION_SCORERS

    started = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores = cross_validate(
            pipeline, X.to_numpy(), y.to_numpy(), cv=cv, scoring=scoring,
            return_train_score=True, n_jobs=1, error_score="raise",
        )
    elapsed = time.perf_counter() - started

    fold_rows = []
    for metric in scoring:
        for fold, value in enumerate(scores[f"test_{metric}"]):
            fold_rows.append(
                {
                    "model": name,
                    "metric": metric,
                    "fold": fold,
                    "test_score": float(value),
                    "train_score": float(scores[f"train_{metric}"][fold]),
                }
            )
    fold_scores = pd.DataFrame(fold_rows)
    summary = (
        fold_scores.groupby("metric")["test_score"]
        .agg(mean="mean", std="std")
        .reset_index()
        .assign(model=name)[["model", "metric", "mean", "std"]]
    )
    _log_cv_summary(name, summary)
    return ModelResult(name=name, cv_summary=summary, fold_scores=fold_scores, fit_seconds=elapsed)


def tune_mlp(
    cfg,
    X: pd.DataFrame,
    y: pd.Series,
    task: str = "classification",
    n_splits: Optional[int] = None,
) -> Tuple[ModelResult, Optional[GridSearchCV]]:
    """GridSearchCV over the configured MLP hyper-parameter grid."""
    logger = get_logger()
    gcfg = cfg.model.grid
    seed = int(cfg.project.seed)

    if task == "classification":
        base = make_mlp_classifier(cfg, seed)
        scoring = str(gcfg.scoring)
    else:
        base = make_mlp_regressor(cfg, seed)
        scoring = "r2"

    pipeline = build_pipeline(base, cfg, task, X.shape[1])
    grid = {}
    for key, value in dict(gcfg.param_grid).items():
        candidates = list(value)
        if key.endswith("hidden_layer_sizes"):
            candidates = [tuple(int(u) for u in (c[0] if isinstance(c, list) and c and isinstance(c[0], list) else c))
                          if isinstance(c, list) else (int(c),) for c in candidates]
        grid[f"model__{key}"] = candidates

    cv = make_cv(cfg, n_splits)
    n_jobs = effective_grid_n_jobs(cfg)
    search = GridSearchCV(
        pipeline,
        param_grid=grid,
        scoring=scoring,
        cv=cv,
        refit=bool(gcfg.refit),
        n_jobs=n_jobs,
        return_train_score=True,
        error_score="raise",
    )
    logger.info("GridSearchCV: %d candidates x %d folds = %d fits (scoring=%s)",
                int(np.prod([len(v) for v in grid.values()])), cv.get_n_splits(), 
                int(np.prod([len(v) for v in grid.values()])) * cv.get_n_splits(), scoring)
    started = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        search.fit(X.to_numpy(), y.to_numpy())
    elapsed = time.perf_counter() - started

    best = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
    logger.info("GridSearchCV best %s = %.4f after %.1fs", scoring, search.best_score_, elapsed)
    for key, value in best.items():
        logger.info("    %-24s = %s", key, value)

    cv_results = pd.DataFrame(search.cv_results_)
    mean_test = cv_results.loc[cv_results["rank_test_score"] == 1, "mean_test_score"].iloc[0]
    std_test = cv_results.loc[cv_results["rank_test_score"] == 1, "std_test_score"].iloc[0]
    summary = pd.DataFrame(
        [{"model": "mlp_tuned", "metric": scoring, "mean": float(mean_test), "std": float(std_test)}]
    )
    result = ModelResult(
        name="mlp_tuned",
        cv_summary=summary,
        fold_scores=pd.DataFrame(),
        fitted_pipeline=search.best_estimator_,
        fit_seconds=elapsed,
        best_params=best,
        extra={"cv_results": cv_results, "best_score": float(search.best_score_)},
    )
    return result, search


def stratified_holdout_split(X: pd.DataFrame, y: pd.Series, cfg, task: str = "classification"):
    """80/20 stratified train/test split with the global seed."""
    test_size = float(
        cfg.dataset.track_a.test_size if task == "classification" else cfg.dataset.track_c.test_size
    )
    stratify = y if task == "classification" and y.nunique() > 1 else None
    return train_test_split(
        X, y, test_size=test_size, random_state=int(cfg.project.seed), stratify=stratify
    )


def learning_curve_history(pipeline) -> pd.DataFrame:
    """Per-epoch training loss and validation score from a fitted MLP pipeline."""
    model = pipeline.named_steps["model"] if isinstance(pipeline, Pipeline) else pipeline
    loss = list(getattr(model, "loss_curve_", []) or [])
    validation = list(getattr(model, "validation_scores_", []) or [])
    n = max(len(loss), len(validation))
    if n == 0:
        return pd.DataFrame(columns=["epoch", "training_loss", "validation_score"])
    return pd.DataFrame(
        {
            "epoch": np.arange(1, n + 1),
            "training_loss": loss + [np.nan] * (n - len(loss)),
            "validation_score": validation + [np.nan] * (n - len(validation)),
        }
    )
