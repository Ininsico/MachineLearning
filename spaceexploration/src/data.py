from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src import config


def load_raw(path=None):
    path = Path(path or config.DATA_PATH)
    df = pd.read_csv(path)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["est_diameter_mean"] = (df["est_diameter_min"] + df["est_diameter_max"]) / 2.0
    df["est_diameter_span"] = df["est_diameter_max"] - df["est_diameter_min"]
    df["sentry_object"] = df["sentry_object"].astype(int)
    return df


def encode_categoricals(X_train, X_val, X_test, cat_cols):
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        le.fit(X_train[col].astype(str))
        encoders[col] = le
        X_train[col] = le.transform(X_train[col].astype(str))
        X_val[col] = le.transform(X_val[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))
    return X_train, X_val, X_test, encoders


def build_pipeline(df: pd.DataFrame):
    df = engineer_features(df)
    feature_cols = config.NUMERIC_COLS + config.CAT_COLS + config.ENGINEERED_COLS

    X = df[feature_cols].copy()
    y = df[config.TARGET_COL].astype(int).values

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=config.VAL_SIZE, stratify=y_trainval, random_state=config.RANDOM_SEED,
    )

    scaler = StandardScaler()
    X_train, X_val, X_test, _ = encode_categoricals(
        X_train, X_val, X_test, config.CAT_COLS
    )
    X_train[config.NUMERIC_COLS + config.ENGINEERED_COLS] = scaler.fit_transform(
        X_train[config.NUMERIC_COLS + config.ENGINEERED_COLS]
    )
    X_val[config.NUMERIC_COLS + config.ENGINEERED_COLS] = scaler.transform(
        X_val[config.NUMERIC_COLS + config.ENGINEERED_COLS]
    )
    X_test[config.NUMERIC_COLS + config.ENGINEERED_COLS] = scaler.transform(
        X_test[config.NUMERIC_COLS + config.ENGINEERED_COLS]
    )

    feature_names = list(X_train.columns)
    splits = {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "scaler": scaler, "feature_names": feature_names,
    }
    return splits


def get_pos_weight(y):
    neg = int((y == 0).sum())
    pos = int((y == 1).sum())
    # weight = neg / pos, standard for BCEWithLogits
    return neg / max(pos, 1)
