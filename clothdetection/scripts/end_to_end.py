import sys, cv2
from pathlib import Path
sys.path.insert(0, '.')

from src.utils.config import cfg
from src.detection.person_detector import PersonDetector
from src.detection.clothing_detector import ClothingDetector
from src.detection.color_analyzer import ColorAnalyzer
from src.recommendation.engine import RecommendationEngine, UserProfile
from src.utils.visualizer import Visualizer

cfg.load()

person_detector = PersonDetector()
clothing_detector = ClothingDetector()
color_analyzer = ColorAnalyzer()
recommender = RecommendationEngine()
viz = Visualizer()

sample_dir = Path('data/samples')
images = sorted(sample_dir.glob('*.jpg')) + sorted(sample_dir.glob('*.jpeg'))

for img_path in images:
    print(f'\n{"="*60}')
    print(f'IMAGE: {img_path.name}')
    print(f'{"="*60}')

    img = cv2.imread(str(img_path))
    if img is None:
        print(f'  Cannot read {img_path.name}')
        continue

    people = person_detector.detect_people(img)
    print(f'  People detected: {len(people)}')

    all_clothing = []
    all_colors = []

    if people:
        for pi, person in enumerate(people):
            px1, py1, px2, py2 = person["bbox"]
            print(f'  Person {pi+1}: bbox=({px1},{py1},{px2},{py2})')

            cropped = person["cropped"]
            items = clothing_detector.detect_clothing(cropped)

            for item in items:
                x1, y1, x2, y2 = item["bbox"]
                item["bbox"] = (x1 + px1, y1 + py1, x2 + px1, y2 + py1)

                region = img[item["bbox"][1]:item["bbox"][3],
                             item["bbox"][0]:item["bbox"][2]]
                colors = color_analyzer.extract_dominant_colors(region, n_colors=3)
                if colors:
                    item["dominant_color"] = colors[0]["name"]
                    all_colors.extend(colors)

                all_clothing.append(item)
                print(f'    {item["category"].upper()}: {item["class"]} ({item["confidence"]:.2f})'
                      f' | Color: {item.get("dominant_color", "?")}')
    else:
        print('  No people detected, classifying full image')
        items = clothing_detector.classify_full_image(img)
        for item in items:
            colors = color_analyzer.extract_dominant_colors(img, n_colors=3)
            if colors:
                item["dominant_color"] = colors[0]["name"]
                all_colors.extend(colors)
            all_clothing.append(item)
            print(f'    {item["category"].upper()}: {item["class"]} ({item["confidence"]:.2f})'
                  f' | Color: {item.get("dominant_color", "?")}')

    unique_colors = list(set(c["name"] for c in all_colors))
    detected_clothing = [c["class"] for c in all_clothing]

    print(f'\n  Dominant colors: {", ".join(unique_colors[:5])}')

    moods = ["casual", "happy", "confident", "cozy", "professional", "energetic", "romantic"]
    weathers = ["mild", "hot", "cold", "rainy"]
    seasons = ["summer", "winter", "spring", "fall"]

    for mood in moods[:2]:
        profile = UserProfile(
            mood=mood,
            weather=weathers[0],
            season=seasons[0],
            detected_colors=unique_colors,
            detected_clothing=detected_clothing
        )

        suggestions = recommender.suggest(profile, n_suggestions=1)
        s = suggestions[0]
        outfit_desc = " + ".join([f"{i['color']} {i['type']}" for i in s.items])
        print(f'\n  [{mood.upper()} RECOMMENDATION]')
        print(f'    Outfit: {outfit_desc}')
        print(f'    Harmony: {s.harmony_score:.2f} | Confidence: {s.confidence:.2f}')
        print(f'    {s.reasoning}')

print('\n=== END-TO-END PIPELINE COMPLETE ===')
