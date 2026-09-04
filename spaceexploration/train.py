from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src import data as data_module
from src import train as train_module
from src import evaluate as eval_module
from src.models import NEOMLP


def run():
    print(f"Device: {config.DEVICE}")
    print("Loading data...")
    df = data_module.load_raw()
    print(f"Loaded {len(df)} rows, {df.shape[1]} columns")

    splits = data_module.build_pipeline(df)
    print(f"Train: {len(splits['X_train'])} | Val: {len(splits['X_val'])} | "
          f"Test: {len(splits['X_test'])}")
    print(f"Positive (hazardous) rate in train: "
          f"{splits['y_train'].mean():.4f}")

    # --- PyTorch MLP on GPU ---
    print("\nTraining PyTorch MLP on GPU...")
    model, history = train_module.train_torch_model(splits)
    mlp_prob = train_module.predict_torch(model, splits["X_test"])
    eval_module.plot_training_history(history, config.MODELS_DIR / "train_history.png")

    # --- Sklearn baselines (CPU) ---
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier

    Xtr = splits["X_train"].values
    ytr = splits["y_train"]
    Xte = splits["X_test"].values
    yte = splits["y_test"]

    weights = {0: 1.0, 1: (yte == 0).sum() / max((yte == 1).sum(), 1)}

    lr = LogisticRegression(max_iter=1000, class_weight=weights)
    lr.fit(Xtr, ytr)
    lr_prob = lr.predict_proba(Xte)[:, 1]

    rf = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                n_jobs=-1, random_state=config.RANDOM_SEED)
    rf.fit(Xtr, ytr)
    rf_prob = rf.predict_proba(Xte)[:, 1]

    # --- Evaluation ---
    # Validation probs for threshold tuning (avoid leakage onto test set)
    mlp_prob_val = train_module.predict_torch(model, splits["X_val"])
    lr_prob_val = lr.predict_proba(splits["X_val"].values)[:, 1]
    rf_prob_val = rf.predict_proba(splits["X_val"].values)[:, 1]

    probs = {
        "PyTorch MLP": mlp_prob,
        "LogisticRegression": lr_prob,
        "RandomForest": rf_prob,
    }
    val_probs = {
        "PyTorch MLP": mlp_prob_val,
        "LogisticRegression": lr_prob_val,
        "RandomForest": rf_prob_val,
    }

    results = {}
    for name, p in probs.items():
        threshold = eval_module.tune_threshold_f1(splits["y_val"], val_probs[name])
        print(f"\nTuned threshold for {name}: {threshold:.3f}")
        results[name] = eval_module.evaluate_model(yte, p, threshold=threshold, name=name)

    eval_module.plot_roc(yte, probs, config.MODELS_DIR / "roc_curve.png")

    # Save the best model (by ROC-AUC) for inference
    import joblib
    best = max(results, key=lambda k: results[k]["roc_auc"])
    if best == "RandomForest":
        joblib.dump(rf, config.MODELS_DIR / "best_model.joblib")
    elif best == "LogisticRegression":
        joblib.dump(lr, config.MODELS_DIR / "best_model.joblib")
    else:
        torch.save(model.state_dict(), config.MODELS_DIR / "best_model.pt")
    print(f"Saved best model '{best}' to {config.MODELS_DIR}")

    # Feature importance from RandomForest
    importance = pd.Series(rf.feature_importances_, index=splits["feature_names"])
    importance.sort_values(ascending=False).to_csv(config.MODELS_DIR / "feature_importance.csv")

    print("\n=== Summary (ROC-AUC) ===")
    for name, r in sorted(results.items(), key=lambda x: -x[1]["roc_auc"]):
        print(f"{name:22s} AUC={r['roc_auc']:.4f}  F1={r['f1']:.4f}  "
              f"Recall={r['recall']:.4f}  Prec={r['precision']:.4f}")

    best = max(results, key=lambda k: results[k]["roc_auc"])
    print(f"\nBest model by ROC-AUC: {best}")
    print(f"Artifacts saved to: {config.MODELS_DIR}")


if __name__ == "__main__":
    run()
