from pathlib import Path
from typing import List, Optional, Union
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

from src.utils.config import cfg


CLOTHING_CLASSES = [
    "t-shirt", "shirt", "blouse", "sweater", "hoodie", "jacket", "coat", "tank_top",
    "jeans", "trousers", "shorts", "skirt", "leggings",
    "dress", "gown", "overalls", "jumpsuit",
    "sneakers", "shoes", "boots", "sandals", "heels",
    "hat", "cap", "glasses", "sunglasses", "scarf", "bag", "backpack", "watch", "belt",
    "necklace", "earrings"
]


class ClothingDetector:
    def __init__(self, classifier_path: str = None, conf_threshold: float = None):
        self.conf_threshold = conf_threshold or cfg.get("yolo", "conf_threshold", default=0.5)
        self.device = cfg.get("yolo", "device", default="cpu")

        classifier_paths = [
            classifier_path,
            str(Path("models/trained/clothing_classifier/weights/best.pt")),
            str(Path("models/trained/clothing_classifier/weights/last.pt")),
        ]

        self.classifier = None
        for cp in classifier_paths:
            if cp and Path(cp).exists():
                self.classifier = YOLO(cp)
                self.classifier.to(self.device)
                break

        if self.classifier is None:
            self.classifier = YOLO("yolov8n-cls.pt")
            self.classifier.to(self.device)

        self.clothing_categories = ["tops", "bottoms", "full_body", "footwear", "accessories"]

    def classify_person_region(self, person_img: np.ndarray) -> dict:
        if person_img.size == 0:
            return {"class": "unknown", "category": "other", "confidence": 0.0}

        person_rgb = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
        results = self.classifier(person_rgb, verbose=False)

        top5 = results[0].probs.top5
        top5conf = results[0].probs.top5conf

        names = self.clothing_categories
        predictions = []
        for idx, conf in zip(top5, top5conf):
            if idx < len(names):
                cls_name = names[idx]
                cat = cls_name
                predictions.append({"class": cls_name, "category": cat, "confidence": float(conf)})

        if not predictions:
            return {"class": "unknown", "category": "other", "confidence": 0.0}

        best = predictions[0]
        return best

    def detect_clothing(self, image: Union[str, np.ndarray, Image.Image]) -> List[dict]:
        if isinstance(image, str):
            img = cv2.imread(image)
        elif isinstance(image, Image.Image):
            img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        else:
            img = image

        h, w = img.shape[:2]

        top_half = img[:h//2, :]
        bottom_half = img[h//2:, :]

        items = []

        top_result = self.classify_person_region(top_half)
        top_result["bbox"] = (0, 0, w, h//2)
        top_result["cropped"] = top_half
        items.append(top_result)

        bottom_result = self.classify_person_region(bottom_half)
        bottom_result["bbox"] = (0, h//2, w, h)
        bottom_result["cropped"] = bottom_half
        items.append(bottom_result)

        return items

    def classify_full_image(self, image: Union[str, np.ndarray]) -> List[dict]:
        if isinstance(image, str):
            img = cv2.imread(image)
        else:
            img = image.copy()

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.classifier(img_rgb, verbose=False)

        top5 = results[0].probs.top5
        top5conf = results[0].probs.top5conf

        items = []
        names = self.clothing_categories
        for idx, conf in zip(top5, top5conf):
            if idx < len(names):
                items.append({
                    "class": names[idx],
                    "category": names[idx],
                    "confidence": float(conf),
                    "bbox": (0, 0, img.shape[1], img.shape[0]),
                    "cropped": img
                })
        return items

    def draw_detections(self, image: Union[str, np.ndarray], items: List[dict]) -> np.ndarray:
        if isinstance(image, str):
            img = cv2.imread(image)
        else:
            img = image.copy()

        color_map = {
            "tops": (255, 0, 0), "bottoms": (0, 255, 0),
            "full_body": (255, 255, 0), "footwear": (0, 255, 255),
            "accessories": (255, 0, 255), "other": (128, 128, 128)
        }

        for item in items:
            x1, y1, x2, y2 = item["bbox"]
            color = color_map.get(item["category"], (128, 128, 128))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = f"{item['class']}: {item['confidence']:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(img, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
            cv2.putText(img, label, (x1 + 5, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        return img


if __name__ == "__main__":
    cfg.load()
    detector = ClothingDetector()
    print("ClothingDetector initialized with classifier")
