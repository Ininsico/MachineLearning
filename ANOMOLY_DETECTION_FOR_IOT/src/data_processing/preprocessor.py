import os
import logging
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

def load_and_preprocess(config) -> dict:

    logger.info(" Loading dataset from: %s", config.RAW_DATA_PATH)

    df = pd.read_csv(config.RAW_DATA_PATH)
    logger.info("   Raw shape: %s", df.shape)

    if config.SAMPLE_SIZE and config.SAMPLE_SIZE < len(df):
        df = df.groupby(config.LABEL_COL, group_keys=False).apply(
            lambda g: g.sample(
                min(len(g), int(config.SAMPLE_SIZE * len(g) / len(df))),
                random_state=config.RANDOM_STATE,
            )
        ).reset_index(drop=True)
        logger.info("   After stratified sampling: %s", df.shape)

    df = df[df[config.LABEL_COL].isin(config.LABEL_MAP.keys())].copy()
    df["binary_label"] = df[config.LABEL_COL].map(config.LABEL_MAP)
    df["attack_type"]  = df[config.LABEL_COL]
    logger.info("   Label distribution:\n%s", df["binary_label"].value_counts().to_string())

    drop_cols = [config.LABEL_COL, "binary_label", "attack_type", "Timestamp"]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols].copy()
    y = df["binary_label"].values
    attack_labels = df["attack_type"].values

    X.replace([np.inf, -np.inf], np.nan, inplace=True)

    thresh = len(X) * 0.5
    X.dropna(axis=1, thresh=thresh, inplace=True)

    X.fillna(X.median(numeric_only=True), inplace=True)
    feature_names = X.columns.tolist()
    logger.info("   Features after cleaning: %d", len(feature_names))

    X_temp, X_test, y_temp, y_test, att_temp, att_test = train_test_split(
        X.values, y, attack_labels,
        test_size=config.TEST_SIZE,
        stratify=y,
        random_state=config.RANDOM_STATE,
    )
    val_ratio = config.VAL_SIZE / (1 - config.TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_ratio,
        stratify=y_temp,
        random_state=config.RANDOM_STATE,
    )
    logger.info("   Train=%d  Val=%d  Test=%d", len(X_train), len(X_val), len(X_test))

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val   = scaler.transform(X_val).astype(np.float32)
    X_test  = scaler.transform(X_test).astype(np.float32)

    os.makedirs(config.PROCESSED_DIR, exist_ok=True)
    joblib.dump(scaler, os.path.join(config.PROCESSED_DIR, "scaler.pkl"))
    np.save(os.path.join(config.PROCESSED_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(config.PROCESSED_DIR, "X_val.npy"),   X_val)
    np.save(os.path.join(config.PROCESSED_DIR, "X_test.npy"),  X_test)
    np.save(os.path.join(config.PROCESSED_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(config.PROCESSED_DIR, "y_val.npy"),   y_val)
    np.save(os.path.join(config.PROCESSED_DIR, "y_test.npy"),  y_test)
    np.save(os.path.join(config.PROCESSED_DIR, "att_test.npy"), att_test, allow_pickle=True)
    pd.Series(feature_names).to_csv(
        os.path.join(config.PROCESSED_DIR, "feature_names.csv"), index=False
    )
    logger.info(" Preprocessing complete. Artifacts saved to: %s", config.PROCESSED_DIR)

    return {
        "X_train":      X_train,
        "X_val":        X_val,
        "X_test":       X_test,
        "y_train":      y_train,
        "y_val":        y_val,
        "y_test":       y_test,
        "att_test":     att_test,
        "scaler":       scaler,
        "feature_names": feature_names,
        "n_features":   len(feature_names),
    }

def load_preprocessed(config) -> dict:

    d = config.PROCESSED_DIR
    logger.info(" Loading preprocessed data from: %s", d)
    return {
        "X_train":       np.load(os.path.join(d, "X_train.npy")),
        "X_val":         np.load(os.path.join(d, "X_val.npy")),
        "X_test":        np.load(os.path.join(d, "X_test.npy")),
        "y_train":       np.load(os.path.join(d, "y_train.npy")),
        "y_val":         np.load(os.path.join(d, "y_val.npy")),
        "y_test":        np.load(os.path.join(d, "y_test.npy")),
        "att_test":      np.load(os.path.join(d, "att_test.npy"), allow_pickle=True),
        "scaler":        joblib.load(os.path.join(d, "scaler.pkl")),
        "feature_names": pd.read_csv(os.path.join(d, "feature_names.csv"))
                           .iloc[:, 0].tolist(),
        "n_features":    len(pd.read_csv(os.path.join(d, "feature_names.csv"))),
    }