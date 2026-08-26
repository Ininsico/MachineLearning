from typing import List, Tuple, Optional
import cv2
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter

from src.utils.config import cfg


class ColorAnalyzer:
    def __init__(self):
        self.color_defs = cfg.get("clothing", "colors", default=[])
        self.color_cache = {c["name"]: np.array(c["rgb"], dtype=np.uint8) for c in self.color_defs}

    def extract_dominant_colors(
        self, image: np.ndarray, n_colors: int = 5
    ) -> List[dict]:
        if image.size == 0:
            return []

        img_small = cv2.resize(image, (100, 100))
        pixels = img_small.reshape(-1, 3)

        kmeans = KMeans(n_clusters=min(n_colors, len(pixels)), random_state=42, n_init=5)
        kmeans.fit(pixels)

        counts = Counter(kmeans.labels_)
        total = sum(counts.values())

        colors = []
        for idx in counts.most_common(n_colors):
            cluster_idx = idx[0]
            count = idx[1]
            rgb = kmeans.cluster_centers_[cluster_idx].astype(int)
            name = self._closest_color_name(rgb)
            hex_color = "#{:02x}{:02x}{:02x}".format(*rgb)
            colors.append({
                "rgb": rgb.tolist(),
                "hex": hex_color,
                "name": name,
                "percentage": round(count / total * 100, 2)
            })

        return colors

    def analyze_region(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> List[dict]:
        x1, y1, x2, y2 = bbox
        region = image[y1:y2, x1:x2]
        if region.size == 0:
            return []
        return self.extract_dominant_colors(region)

    def _closest_color_name(self, rgb: np.ndarray) -> str:
        min_dist = float("inf")
        closest = "Unknown"
        for name, ref_rgb in self.color_cache.items():
            dist = np.linalg.norm(rgb.astype(float) - ref_rgb.astype(float))
            if dist < min_dist:
                min_dist = dist
                closest = name
        return closest

    def get_color_harmony_score(self, colors: List[str]) -> float:
        if len(colors) < 2:
            return 1.0

        harmonies = cfg.get("recommendation", "color_harmony", default={})
        score = 0.0
        pairs = 0

        for i, c1 in enumerate(colors):
            for c2 in colors[i + 1:]:
                pairs += 1
                is_complementary = any(
                    c1 in pair and c2 in pair
                    for pair in harmonies.get("complementary", [])
                )
                is_analogous = any(
                    c1 in group and c2 in group
                    for group in harmonies.get("analogous", [])
                )
                is_mono = any(
                    c1 in group and c2 in group
                    for group in harmonies.get("monochromatic", [])
                )
                if is_complementary:
                    score += 1.0
                elif is_analogous:
                    score += 0.8
                elif is_mono:
                    score += 0.6
                else:
                    score += 0.3

        return score / pairs if pairs > 0 else 1.0


if __name__ == "__main__":
    cfg.load()
    analyzer = ColorAnalyzer()
    test_img = cv2.imread("data/samples/test_clothing.jpg")
    if test_img is not None:
        colors = analyzer.extract_dominant_colors(test_img)
        print(f"Dominant colors: {[c['name'] for c in colors]}")
