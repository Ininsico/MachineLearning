import sys, cv2
sys.path.insert(0, '.')
from src.detection.person_detector import PersonDetector
from src.detection.clothing_detector import ClothingDetector
from src.detection.color_analyzer import ColorAnalyzer

pd = PersonDetector()
cd = ClothingDetector()
ca = ColorAnalyzer()

img = cv2.imread('data/samples/person_casual.jpg')
if img is None:
    print('No test image found')
else:
    people = pd.detect_people(img)
    print(f'People: {len(people)}')
    for p in people:
        items = cd.detect_clothing(p['cropped'])
        for i in items:
            print(f'  {i["category"]}: {i["class"]} ({i["confidence"]:.2f})')
    print('No verbose YOLO spam - clean output!')
