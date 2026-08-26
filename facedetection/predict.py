import cv2
import numpy as np
import json
import pickle
from pathlib import Path
from skimage.feature import hog
import argparse


def load_model(model_dir: str):
    model_path = Path(model_dir)
    with open(model_path / "svm_pipeline.pkl", "rb") as f:
        pipeline = pickle.load(f)
    with open(model_path / "label_map.json") as f:
        label_map = json.load(f)
    with open(model_path / "config.json") as f:
        config = json.load(f)
    labels_rev = {v: k for k, v in label_map.items()}
    return pipeline, label_map, labels_rev, config


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


def detect_faces(img: np.ndarray, min_face_size: int = 60) -> tuple:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(min_face_size, min_face_size)
    )
    return faces, gray


def predict_image(
    image_path: str,
    pipeline,
    labels_rev: dict,
    config: dict,
    confidence_threshold: float = 0.6,
    display: bool = False,
):
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Could not read image: {image_path}")
        return []

    target_size = tuple(config.get("target_size", [128, 128]))
    faces, gray = detect_faces(img)
    results = []

    for x, y, w, h in faces:
        face_roi = gray[y : y + h, x : x + w]
        face_roi = cv2.resize(face_roi, target_size, interpolation=cv2.INTER_LANCZOS4)

        hog_feat = extract_hog_features(face_roi).reshape(1, -1)
        probs = pipeline.predict_proba(hog_feat)[0]
        pred_idx = np.argmax(probs)
        confidence = float(probs[pred_idx])
        name = labels_rev.get(pred_idx, "unknown")

        results.append({
            "name": name,
            "confidence": round(confidence * 100, 1),
            "bbox": [int(x), int(y), int(w), int(h)],
        })

        if display:
            color = (0, 255, 0) if confidence >= confidence_threshold else (0, 0, 255)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            label = f"{name} ({confidence*100:.0f}%)"
            cv2.putText(
                img, label, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
            )

    if display:
        cv2.imshow("Face Recognition", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return results


def predict_video(
    video_path: str,
    pipeline,
    labels_rev: dict,
    config: dict,
    confidence_threshold: float = 0.6,
    display: bool = True,
):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return

    target_size = tuple(config.get("target_size", [128, 128]))
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    print(f"[*] Processing video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"    FPS: {fps:.1f}, Frames: {frame_count}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))

        for x, y, w, h in faces:
            face_roi = gray[y : y + h, x : x + w]
            face_roi = cv2.resize(face_roi, target_size, interpolation=cv2.INTER_LANCZOS4)
            hog_feat = extract_hog_features(face_roi).reshape(1, -1)
            probs = pipeline.predict_proba(hog_feat)[0]
            pred_idx = np.argmax(probs)
            confidence = float(probs[pred_idx])
            name = labels_rev.get(pred_idx, "unknown")

            color = (0, 255, 0) if confidence >= confidence_threshold else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                frame, f"{name} ({confidence*100:.0f}%)",
                (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
            )

        if display:
            cv2.imshow("Face Recognition - Video", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recognize faces in images/video using HOG + SVM")
    parser.add_argument("input", help="Path to image, video, or directory")
    parser.add_argument("--model", "-m", default="model", help="Model directory (default: model)")
    parser.add_argument("--threshold", "-t", type=float, default=0.6, help="Confidence threshold (0-1)")
    parser.add_argument("--display", "-d", action="store_true", help="Display results with OpenCV GUI")
    args = parser.parse_args()

    pipeline, label_map, labels_rev, config = load_model(args.model)
    print(f"[*] Loaded model with {len(label_map)} classes: {list(label_map.keys())}")

    input_path = Path(args.input)
    if input_path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
        predict_video(input_path, pipeline, labels_rev, config, args.threshold, args.display)
    elif input_path.is_dir():
        for img_path in sorted(input_path.iterdir()):
            if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                results = predict_image(str(img_path), pipeline, labels_rev, config, args.threshold, args.display)
                for r in results:
                    print(f"  {img_path.name}: {r['name']} ({r['confidence']}%)")
    else:
        results = predict_image(str(input_path), pipeline, labels_rev, config, args.threshold, args.display)
        for r in results:
            print(f"  {input_path.name}: {r['name']} ({r['confidence']}%)")
        if not results:
            print("  No faces detected.")
