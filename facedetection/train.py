import cv2
import numpy as np
import json
import pickle
from pathlib import Path
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
import argparse
from skimage.feature import hog


def extract_hog_features(img_gray: np.ndarray) -> np.ndarray:
    features = hog(
        img_gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )
    return features


def load_data(data_dir: str, target_size: tuple = (128, 128)):
    data_path = Path(data_dir)
    X, y = [], []
    labels = []

    for split in ["train", "val"]:
        split_path = data_path / split
        if not split_path.exists():
            continue

        for person_dir in sorted(split_path.iterdir()):
            if not person_dir.is_dir():
                continue
            person_name = person_dir.name

            for img_path in person_dir.iterdir():
                if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                    continue
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = cv2.resize(img, target_size, interpolation=cv2.INTER_LANCZOS4)

                hog_feat = extract_hog_features(img)
                X.append(hog_feat)
                y.append(person_name)

    return np.array(X), np.array(y)


def train_svm(data_dir: str, output_dir: str, target_size: tuple = (128, 128)):
    print("[*] Loading training data...")
    X, y = load_data(data_dir, target_size)

    if len(X) == 0:
        print("[!] No training data found. Run collect.py first.")
        return

    unique_classes = np.unique(y)
    print(f"    Classes: {list(unique_classes)}")
    print(f"    Samples: {len(X)}")
    print(f"    Feature dim: {X.shape[1]}")

    print("[*] Training SVM classifier...")
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=42)),
    ])
    pipeline.fit(X, y)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    with open(out_path / "svm_pipeline.pkl", "wb") as f:
        pickle.dump(pipeline, f)

    label_map = {name: idx for idx, name in enumerate(unique_classes)}
    with open(out_path / "label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)

    # Save config
    config = {"target_size": list(target_size), "hog_params": {"orientations": 9, "pixels_per_cell": [8, 8]}}
    with open(out_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"[*] Model saved to {out_path / 'svm_pipeline.pkl'}")
    print(f"[*] Label map saved to {out_path / 'label_map.json'}")

    # Validation accuracy
    val_path = Path(data_dir) / "val"
    if val_path.exists():
        print("\n[*] Evaluating on validation set...")
        X_val, y_val = load_data(data_dir.replace("train", "val"), target_size) if "train" in str(data_dir) else ([], [])
        # Proper val loading
        X_val, y_val = [], []
        for person_dir in sorted(val_path.iterdir()):
            if not person_dir.is_dir():
                continue
            person_name = person_dir.name
            for img_path in person_dir.iterdir():
                if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                    continue
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = cv2.resize(img, target_size, interpolation=cv2.INTER_LANCZOS4)
                X_val.append(extract_hog_features(img))
                y_val.append(person_name)

        if len(X_val) > 0:
            X_val = np.array(X_val)
            y_pred = pipeline.predict(X_val)
            acc = np.mean(y_pred == y_val) * 100
            print(f"    Validation accuracy: {acc:.1f}% ({len(X_val)} samples)")

    return pipeline, label_map


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train face recognition model (HOG + SVM)")
    parser.add_argument("--data", "-d", default="data", help="Data directory (default: data)")
    parser.add_argument("--output", "-o", default="model", help="Output directory (default: model)")
    args = parser.parse_args()

    train_svm(Path(args.data), args.output)
