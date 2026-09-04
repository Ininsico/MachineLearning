"""Inference entry point for hazardous NEO prediction.

Usage:
    python predict.py data/neo.csv            # score a CSV of NEO records
    python predict.py --name "2023 XY" --est_diameter_min 0.3 --est_diameter_max 0.7 \
        --relative_velocity 60000 --miss_distance 4000000 --absolute_magnitude 19 \
        --orbiting_body Earth --sentry_object False
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src import data as data_module


def load_model():
    import joblib
    model_path = config.MODELS_DIR / "best_model.joblib"
    if model_path.exists():
        return joblib.load(model_path), "sklearn"
    import torch
    from src.models import NEOMLP
    model = NEOMLP(n_features=len(config.NUMERIC_COLS + config.CAT_COLS + config.ENGINEERED_COLS))
    model.load_state_dict(torch.load(config.MODELS_DIR / "best_model.pt", weights_only=True))
    model.eval()
    return model, "torch"


def build_features_from_args(args):
    row = {
        "est_diameter_min": float(args.est_diameter_min),
        "est_diameter_max": float(args.est_diameter_max),
        "relative_velocity": float(args.relative_velocity),
        "miss_distance": float(args.miss_distance),
        "absolute_magnitude": float(args.absolute_magnitude),
        "orbiting_body": args.orbiting_body,
        "sentry_object": args.sentry_object.lower() in ("1", "true", "yes"),
    }
    return pd.DataFrame([row])


def score(model, kind, X):
    if kind == "sklearn":
        proba = model.predict_proba(X.values)[:, 1]
    else:
        import torch
        with torch.no_grad():
            proba = torch.sigmoid(model(torch.tensor(X.values, dtype=torch.float32))).numpy()
    return proba


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Predict hazardous NEOs")
    parser.add_argument("csv", nargs="?", help="CSV of NEO records to score")
    parser.add_argument("--name")
    parser.add_argument("--est_diameter_min", default=0.0)
    parser.add_argument("--est_diameter_max", default=0.0)
    parser.add_argument("--relative_velocity", default=0.0)
    parser.add_argument("--miss_distance", default=0.0)
    parser.add_argument("--absolute_magnitude", default=0.0)
    parser.add_argument("--orbiting_body", default="Earth")
    parser.add_argument("--sentry_object", default="False")
    args = parser.parse_args()

    model, kind = load_model()

    if args.csv:
        raw = data_module.load_raw(args.csv)
        df = data_module.engineer_features(raw)
        X = df[config.NUMERIC_COLS + config.CAT_COLS + config.ENGINEERED_COLS].copy()
        X, _, _, _ = data_module.encode_categoricals(X, X.copy(), X.copy(), config.CAT_COLS)
        proba = score(model, kind, X)
        out = raw[["id", "name"]].copy() if "name" in raw else pd.DataFrame(index=range(len(X)))
        out["hazardous_probability"] = np.round(proba, 4)
        out["predicted_hazardous"] = (proba >= 0.5).astype(bool)
        print(out.to_string(index=False))
    else:
        X = data_module.engineer_features(build_features_from_args(args))
        X = X[config.NUMERIC_COLS + config.CAT_COLS + config.ENGINEERED_COLS].copy()
        X, _, _, _ = data_module.encode_categoricals(X, X.copy(), X.copy(), config.CAT_COLS)
        proba = score(model, kind, X)[0]
        print(f"Hazardous probability: {proba:.4f} -> {'HAZARDOUS' if proba >= 0.5 else 'not hazardous'}")


if __name__ == "__main__":
    main()
