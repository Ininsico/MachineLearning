"""Optional PyTorch backend for the MLP, with CUDA support.

WHY THIS EXISTS
---------------
The study specification mandates scikit-learn's ``MLPClassifier``/``MLPRegressor``,
and that remains the default backend. This module adds an *opt-in* equivalent so the
same architecture can run on a GPU.

Honest performance note for the current dataset: Track A is 144 genera x 254
features with a (64, 32) network, i.e. roughly 18k parameters and ~2 MFLOP per
forward pass. A grid search of a few hundred such fits is dominated by kernel
launch overhead on any GPU - and the available GTX 1050 Ti Max-Q has only 6 SMs.
For this size, CPU training is faster. The GPU backend becomes worthwhile when the
problem grows: ASV-level (rather than genus-level) features, PICRUSt2 functional
gene abundances, many more studies, or wider/deeper networks.

``TorchMLPClassifier`` implements the scikit-learn estimator contract
(``get_params``/``set_params``/``fit``/``predict``/``predict_proba``) so it can be
dropped into the existing ``Pipeline``, ``GridSearchCV``, ``cross_validate`` and
plotting code with no other changes. It exposes ``loss_curve_`` and
``validation_scores_`` so the per-epoch learning-curve machinery works identically.
"""

from __future__ import annotations

import os
from typing import Optional, Sequence, Tuple

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from .logging_utils import get_logger

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    TORCH_AVAILABLE = False


def resolve_device(preference: str = "auto") -> str:
    preference = (preference or "auto").strip().lower()
    if not TORCH_AVAILABLE:
        return "cpu"
    if preference == "cpu":
        return "cpu"
    if preference in {"auto", "cuda"} and torch.cuda.is_available():
        return "cuda"
    if preference == "cuda":
        get_logger().warning("CUDA requested but unavailable; falling back to CPU")
    return "cpu"


def describe_device(device: str) -> str:
    if device == "cuda" and TORCH_AVAILABLE and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        return f"cuda ({props.name}, {props.multi_processor_count} SMs, sm_{props.major}{props.minor})"
    return "cpu"


def seed_everything(seed: int, deterministic: bool = True) -> None:
    if not TORCH_AVAILABLE:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:  # pragma: no cover - older torch builds
            pass


def _build_network(input_dim: int, hidden: Sequence[int], activation: str, dropout: float):
    layers = []
    previous = input_dim
    for width in hidden:
        layers.append(nn.Linear(previous, int(width)))
        layers.append(nn.ReLU() if activation == "relu" else nn.Tanh())
        if dropout > 0:
            layers.append(nn.Dropout(float(dropout)))
        previous = int(width)
    layers.append(nn.Linear(previous, 1))
    return nn.Sequential(*layers)


class TorchMLPClassifier(ClassifierMixin, BaseEstimator):
    """Binary MLP classifier with an interface matching ``MLPClassifier``.

    Mixin order matters: scikit-learn requires ``ClassifierMixin`` to appear *before*
    ``BaseEstimator`` in the MRO, because tag resolution walks the inheritance chain
    with ``super()``. Declaring ``(BaseEstimator, ClassifierMixin)`` leaves
    ``estimator_type`` unset, so scikit-learn does not recognise the estimator as a
    classifier and hands the full ``(n, 2)`` probability matrix to scorers such as
    ``roc_auc``, which then fails with "y should be a 1d array". The order below is
    the documented requirement.

    Parameters mirror the study specification; ``L2`` regularisation is applied as
    Adam ``weight_decay`` (equivalent to ``alpha`` in scikit-learn's MLP) and
    ``early_stopping`` uses a held-out ``validation_fraction`` of the training data.
    """

    def __init__(
        self,
        hidden_layer_sizes: Tuple[int, ...] = (64, 32),
        activation: str = "relu",
        alpha: float = 0.001,
        learning_rate_init: float = 0.001,
        batch_size: int = 32,
        max_iter: int = 1000,
        early_stopping: bool = True,
        validation_fraction: float = 0.15,
        n_iter_no_change: int = 15,
        tol: float = 1e-4,
        dropout: float = 0.0,
        class_weight: Optional[str] = None,
        device: str = "auto",
        deterministic: bool = True,
        random_state: Optional[int] = 42,
        verbose: bool = False,
    ) -> None:
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
        self.dropout = dropout
        self.class_weight = class_weight
        self.device = device
        self.deterministic = deterministic
        self.random_state = random_state
        self.verbose = verbose

    def _make_tensor(self, array) -> "torch.Tensor":
        return torch.as_tensor(np.asarray(array, dtype=np.float32), device=self._device)

    def fit(self, X, y):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not installed; use model.backend: sklearn")
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features
        self.classes_ = np.array([0, 1])

        seed = int(self.random_state if self.random_state is not None else 42)
        seed_everything(seed, deterministic=bool(self.deterministic))
        self._device = resolve_device(self.device)

        generator = torch.Generator().manual_seed(seed)
        if self.early_stopping and self.validation_fraction and self.validation_fraction > 0:
            n_val = max(1, int(round(n_samples * float(self.validation_fraction))))
            permuted = torch.randperm(n_samples, generator=generator)
            val_idx, train_idx = permuted[:n_val], permuted[n_val:]
        else:
            train_idx = torch.arange(n_samples)
            val_idx = torch.empty(0, dtype=torch.long)

        X_train = self._make_tensor(X[train_idx.numpy()])
        y_train = self._make_tensor(y[train_idx.numpy()])
        X_val = self._make_tensor(X[val_idx.numpy()]) if len(val_idx) else None
        y_val = self._make_tensor(y[val_idx.numpy()]) if len(val_idx) else None

        hidden = self.hidden_layer_sizes
        if isinstance(hidden, (int, np.integer)):
            hidden = (int(hidden),)
        model = _build_network(n_features, tuple(hidden), str(self.activation), float(self.dropout))
        model = model.to(self._device)

        pos_weight = None
        if str(self.class_weight).lower() == "balanced":
            n_pos = float((y_train == 1).sum().item())
            n_neg = float((y_train == 0).sum().item())
            if n_pos > 0:
                pos_weight = torch.tensor([n_neg / n_pos], device=self._device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimiser = torch.optim.Adam(
            model.parameters(), lr=float(self.learning_rate_init), weight_decay=float(self.alpha)
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser, mode="min", factor=0.5, patience=max(3, self.n_iter_no_change // 3)
        )

        batch_size = int(self.batch_size) if self.batch_size and self.batch_size > 0 else n_samples
        self.loss_curve_ = []
        self.validation_scores_ = []
        best_loss = np.inf
        best_state = None
        epochs_without_improvement = 0

        for epoch in range(1, int(self.max_iter) + 1):
            model.train()
            permutation = torch.randperm(len(train_idx), generator=generator)
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, len(permutation), batch_size):
                batch = permutation[start:start + batch_size]
                optimiser.zero_grad()
                logits = model(X_train[batch]).squeeze(-1)
                loss = criterion(logits, y_train[batch])
                loss.backward()
                optimiser.step()
                epoch_loss += float(loss.item())
                n_batches += 1

            train_loss = epoch_loss / max(1, n_batches)
            self.loss_curve_.append(train_loss)

            if X_val is not None:
                model.eval()
                with torch.no_grad():
                    val_logits = model(X_val).squeeze(-1)
                    val_loss = float(criterion(val_logits, y_val).item())
                    predictions = (torch.sigmoid(val_logits) >= 0.5).float()
                    val_score = float((predictions == y_val).float().mean().item())
                self.validation_scores_.append(val_score)
                scheduler.step(val_loss)
                monitored = val_loss
            else:
                monitored = train_loss

            if monitored < best_loss - float(self.tol):
                best_loss = monitored
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if self.verbose and epoch % 25 == 0:
                get_logger().info("    epoch %4d  train_loss=%.5f  monitored=%.5f", epoch, train_loss, monitored)

            # The tolerance/patience criterion applies whether or not early stopping
            # is enabled, mirroring scikit-learn's MLPClassifier: with early stopping
            # it monitors a held-out split, otherwise it monitors the training loss.
            # Gating this on `early_stopping` alone would force every fit to run the
            # full max_iter budget, which made cross-validation and the grid search
            # roughly 20x slower than they needed to be.
            if epochs_without_improvement >= int(self.n_iter_no_change):
                if self.verbose:
                    get_logger().info(
                        "    stopped at epoch %d (%s plateaued)",
                        epoch, "validation loss" if self.early_stopping else "training loss",
                    )
                break

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        self._model = model
        self.n_iter_ = len(self.loss_curve_)
        self.best_loss_ = float(best_loss)
        return self

    def _logits(self, X) -> "torch.Tensor":
        check_is_fitted(self, "_model")
        self._model.eval()
        with torch.no_grad():
            return self._model(self._make_tensor(np.asarray(X, dtype=np.float32))).squeeze(-1)

    def predict_proba(self, X) -> np.ndarray:
        probabilities = torch.sigmoid(self._logits(X)).cpu().numpy().reshape(-1)
        return np.column_stack([1.0 - probabilities, probabilities])

    def predict(self, X) -> np.ndarray:
        return (torch.sigmoid(self._logits(X)).cpu().numpy().reshape(-1) >= 0.5).astype(int)
