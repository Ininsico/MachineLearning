from pathlib import Path
from typing import List, Optional, Union
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

from src.utils.config import cfg


class PersonDetector:
    def __init__(self, model_path: str = None, conf_threshold: float = None):
        if model_path is None:
            model_path = cfg.get("yolo", "model", default="yolov8n.pt")
        self.conf_threshold = conf_threshold or cfg.get("yolo", "conf_threshold", default=0.5)
        self.device = cfg.get("yolo", "device", default="cpu")

        model_file = Path(model_path)
        if model_file.exists():
            self.model = YOLO(str(model_file))
        else:
            self.model = YOLO(model_path)

        self.model.to(self.device)

    def detect_people(self, image: Union[str, np.ndarray, Image.Image]) -> List[dict]:
        if isinstance(image, str):
            img = cv2.imread(image)
        elif isinstance(image, Image.Image):
            img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        else:
            img = image

        results = self.model(img, conf=self.conf_threshold, verbose=False)[0]

        people = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                cropped = img[y1:y2, x1:x2]
                people.append({
                    "bbox": (x1, y1, x2, y2),
                    "confidence": confidence,
                    "cropped": cropped
                })

        return people

    def draw_boxes(self, image: Union[str, np.ndarray], people: List[dict]) -> np.ndarray:
        if isinstance(image, str):
            img = cv2.imread(image)
        else:
            img = image.copy()

        for person in people:
            x1, y1, x2, y2 = person["bbox"]
            conf = person["confidence"]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Person: {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(img, (x1, y1 - th - 10), (x1 + tw + 10, y1), (0, 255, 0), -1)
            cv2.putText(img, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        return img


if __name__ == "__main__":
    cfg.load()
    detector = PersonDetector()
    print("PersonDetector initialized with YOLOv8")
    test_img = "data/samples/test_person.jpg"
    if Path(test_img).exists():
        people = detector.detect_people(test_img)
        print(f"Detected {len(people)} people")
