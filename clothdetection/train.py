import sys
from pathlib import Path
import argparse
from ultralytics import YOLO

from src.utils.config import cfg


def train_clothing_model(
    model_name: str = "yolov8n.pt",
    data_yaml: str = "datasets/clothing/dataset.yaml",
    epochs: int = None,
    batch: int = None,
    lr: float = None,
    imgsz: int = None,
    device: str = None,
    augment: bool = None,
    output_dir: str = None
):
    cfg.load()

    epochs = epochs or cfg.get("yolo", "epochs", default=50)
    batch = batch or cfg.get("yolo", "batch", default=16)
    lr = lr or cfg.get("yolo", "lr", default=0.001)
    imgsz = imgsz or cfg.get("yolo", "imgsz", default=640)
    device = device or cfg.get("yolo", "device", default="cpu")
    augment = augment if augment is not None else cfg.get("yolo", "augment", default=True)
    output_dir = output_dir or str(cfg.get_path("trained_models"))

    print("=" * 60)
    print("  CLOTHMIND AI - YOLO Clothing Detection Training")
    print("=" * 60)
    print(f"  Model:      {model_name}")
    print(f"  Data:       {data_yaml}")
    print(f"  Epochs:     {epochs}")
    print(f"  Batch:      {batch}")
    print(f"  Learning:   {lr}")
    print(f"  Image Size: {imgsz}")
    print(f"  Device:     {device}")
    print(f"  Augment:    {augment}")
    print(f"  Output:     {output_dir}")
    print("=" * 60)

    if not Path(data_yaml).exists():
        print(f"ERROR: Dataset yaml not found at {data_yaml}")
        print("Run data preprocessing first: python -m src.data.downloader")
        sys.exit(1)

    model = YOLO(model_name)

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch,
        lr0=lr,
        imgsz=imgsz,
        device=device,
        augment=augment,
        project=output_dir,
        name="clothing_yolo",
        exist_ok=True,
        patience=15,
        save=True,
        save_period=10,
        plots=True,
        val=True,
    )

    export_path = model.export(format="onnx")
    print(f"\nModel exported to ONNX: {export_path}")

    final_path = Path(output_dir) / "clothing_yolo" / "weights" / "best.pt"
    if final_path.exists():
        print(f"Trained model saved: {final_path}")

    return results


def validate_model(model_path: str, data_yaml: str = "datasets/clothing/dataset.yaml"):
    print(f"\nValidating model: {model_path}")
    model = YOLO(model_path)
    metrics = model.val(data=data_yaml)
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLO clothing detection model")
    parser.add_argument("--model", default="yolov8n.pt", help="Base YOLO model")
    parser.add_argument("--data", default="datasets/clothing/dataset.yaml", help="Dataset config")
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs")
    parser.add_argument("--batch", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--imgsz", type=int, default=None, help="Image size")
    parser.add_argument("--device", default=None, help="Device (cpu/cuda)")
    parser.add_argument("--validate", action="store_true", help="Validate after training")

    args = parser.parse_args()

    results = train_clothing_model(
        model_name=args.model,
        data_yaml=args.data,
        epochs=args.epochs,
        batch=args.batch,
        lr=args.lr,
        imgsz=args.imgsz,
        device=args.device,
    )

    if args.validate:
        final_path = Path("models/trained/clothing_yolo/weights/best.pt")
        if final_path.exists():
            validate_model(str(final_path), args.data)

    print("\nTraining complete!")
