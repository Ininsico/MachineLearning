from pathlib import Path
from typing import List, Optional
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from src.utils.config import cfg


class Visualizer:
    @staticmethod
    def draw_detection_results(
        image: np.ndarray,
        people: List[dict],
        clothing: List[dict],
        colors: List[dict] = None
    ) -> np.ndarray:
        img = image.copy()
        h, w = img.shape[:2]

        info_panel = np.ones((160, 350, 3), dtype=np.uint8) * 240
        y_offset = 15

        cv2.putText(info_panel, "CLOTHING DETECTION", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        y_offset += 30

        if people:
            cv2.putText(info_panel, f"People detected: {len(people)}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            y_offset += 25

        for item in clothing:
            color_swatch = np.array(cfg.get("clothing", "colors", default=[]))
            label = f"  {item['class']} ({item['confidence']:.2f})"
            cv2.putText(info_panel, label, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
            y_offset += 20

        img[10:170, 10:360] = cv2.addWeighted(
            img[10:170, 10:360], 0.3, info_panel, 0.7, 0
        )

        color_map = {
            "tops": (255, 0, 0), "bottoms": (0, 255, 0),
            "full_body": (255, 255, 0), "footwear": (0, 255, 255),
            "accessories": (255, 0, 255), "other": (128, 128, 128)
        }

        for item in clothing:
            x1, y1, x2, y2 = item["bbox"]
            color = color_map.get(item["category"], (128, 128, 128))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = f"{item['class']}"
            cv2.putText(img, label, (x1 + 5, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        for person in people:
            x1, y1, x2, y2 = person["bbox"]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"Person", (x1 + 5, y2 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if colors:
            cx, cy = w - 180, 20
            cv2.rectangle(img, (cx - 5, cy - 5), (cx + 170, cy + 110), (255, 255, 255), -1)
            cv2.putText(img, "COLORS", (cx + 5, cy + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            for i, c in enumerate(colors[:4]):
                color_rgb = tuple(c.get("rgb", [128, 128, 128]))
                cv2.rectangle(img, (cx + 5, cy + 25 + i * 20),
                              (cx + 25, cy + 40 + i * 20), color_rgb, -1)
                cv2.putText(img, f"{c['name']} {c['percentage']:.0f}%",
                            (cx + 30, cy + 37 + i * 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)

        return img

    @staticmethod
    def show_outfit_suggestion(suggestion, save_path: str = None):
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis("off")
        ax.set_facecolor("#f5f5f5")

        title = f"Outfit Suggestion - {suggestion.style.upper()} | {suggestion.vibe.upper()}"
        ax.text(5, 9.5, title, ha="center", va="center",
                fontsize=14, fontweight="bold", fontfamily="serif")

        y = 8.0
        for item in suggestion.items:
            color = item.get("color", "Black")
            item_type = item.get("type", "Unknown")
            category = item.get("category", "")

            color_hex = "#808080"
            for cd in cfg.get("clothing", "colors", default=[]):
                if cd["name"] == color:
                    color_hex = cd["hex"]
                    break

            rect = FancyBboxPatch(
                (1.5, y - 0.6), 7, 0.5,
                boxstyle="round,pad=0.1",
                facecolor=color_hex, edgecolor="#333333", linewidth=1.5
            )
            ax.add_patch(rect)

            text_color = "white" if color in ["Black", "Navy", "Maroon"] else "black"
            ax.text(5, y - 0.35, f"{color} {item_type.upper()}",
                    ha="center", va="center", fontsize=10,
                    fontweight="bold", color=text_color)
            y -= 0.9

        details = (
            f"Harmony Score: {suggestion.harmony_score:.2f}\n"
            f"Confidence: {suggestion.confidence:.2f}\n"
            f"Mood: {suggestion.vibe.capitalize()}"
        )
        ax.text(9.5, 4, details, ha="right", va="center",
                fontsize=8, fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="gray"))

        ax.text(5, 0.5, suggestion.reasoning, ha="center", va="center",
                fontsize=9, style="italic", wrap=True)

        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved: {save_path}")

        return fig


if __name__ == "__main__":
    cfg.load()
    print("Visualizer ready")
