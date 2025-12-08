# OFFICE_DETECTOR_SIMPLE.py
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import time
import os

print("🔥 LOADING OFFICE DETECTOR...")

# Use CPU only to save memory
device = 'cpu'
torch.set_num_threads(4)  # Limit CPU threads

# Load only ONE good model
print("Loading YOLOv8n (lightweight but decent)...")
model = YOLO('yolov8n.pt')  # Smallest model that works

# Try to load your custom office model
office_model = None
office_labels = {}

# Look for your model
model_files = ['office_data', 'office_data.pt', 'best.pt', 'office.pt']
for mfile in model_files:
    if os.path.exists(mfile):
        try:
            office_model = YOLO(mfile)
            print(f"✅ Loaded office model: {mfile}")
            
            # Try to get class names
            if hasattr(office_model, 'names'):
                office_labels = office_model.names
                print(f"Office model classes: {list(office_labels.values())[:10]}...")
            break
        except:
            continue

if office_model is None:
    print("⚠️ No custom office model found, using YOLOv8n only")

print("\n" + "="*60)
print("OFFICE OBJECT DETECTOR - READY!")
print("="*60)

# COCO classes (YOLOv8n)
COCO_CLASSES = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
    5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
    10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
    14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
    20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
    25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
    30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
    35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket',
    39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife',
    44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich',
    49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza',
    54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant',
    59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop',
    64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave',
    69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book',
    74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier',
    79: 'toothbrush'
}

# Office-specific objects to focus on
OFFICE_OBJECTS = ['person', 'chair', 'laptop', 'mouse', 'keyboard', 'cell phone', 
                  'book', 'bottle', 'cup', 'dining table', 'tv', 'clock', 'backpack',
                  'handbag', 'suitcase']

def detect_objects(image):
    """Detect objects in image"""
    results = []
    
    # Run YOLOv8n (always available)
    try:
        yolo_results = model(image, conf=0.25, verbose=False, device=device)[0]
        
        if yolo_results.boxes is not None:
            for box in yolo_results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                
                
                label = COCO_CLASSES.get(cls_id, f"object_{cls_id}")
                if label in OFFICE_OBJECTS and conf > 0.3:
                    results.append({
                        'label': label,
                        'box': (x1, y1, x2, y2),
                        'confidence': conf,
                        'source': 'yolov8n'
                    })
    except Exception as e:
        print(f"YOLO error: {e}")
    
    
    if office_model is not None:
        try:
            office_results = office_model(image, conf=0.25, verbose=False, device=device)[0]
            
            if office_results.boxes is not None:
                for box in office_results.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    
                    label = office_labels.get(cls_id, f"office_obj_{cls_id}")
                    if conf > 0.3:
                        results.append({
                            'label': label,
                            'box': (x1, y1, x2, y2),
                            'confidence': conf,
                            'source': 'office_model'
                        })
        except Exception as e:
            print(f"Office model error: {e}")
    
    return results

def draw_detections(image, detections):
    """Draw detections on image"""
    output = image.copy()
    
    colors = {
        'yolov8n': (0, 255, 0),      # Green
        'office_model': (255, 0, 0),  # Blue
    }
    
    for det in detections:
        x1, y1, x2, y2 = det['box']
        color = colors.get(det['source'], (255, 255, 255))
        
        # Draw box
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label_text = f"{det['label']} ({det['confidence']:.2f})"
        cv2.putText(output, label_text, (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return output

# ========== MAIN ==========
print("\n📸 STARTING DETECTION...")
print("Press 'q' to quit, 's' to save image")

# Try webcam first
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("⚠️ Webcam not available, using test image...")
    # Create test office image
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Draw office scene
    cv2.rectangle(img, (100, 100), (300, 400), (200, 200, 200), -1)  # Person
    cv2.rectangle(img, (350, 200), (500, 350), (150, 100, 50), -1)   # Desk
    cv2.rectangle(img, (400, 220), (450, 270), (100, 100, 200), -1)  # Laptop
    cv2.rectangle(img, (370, 300), (400, 320), (50, 50, 50), -1)     # Phone
    cv2.rectangle(img, (200, 300), (250, 350), (200, 150, 100), -1)  # Chair
    cv2.rectangle(img, (50, 250), (120, 320), (100, 100, 255), -1)   # Bottle
    
    cv2.putText(img, "OFFICE TEST SCENE", (200, 50),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    test_mode = True
else:
    test_mode = False

frame_count = 0
fps = 0
fps_time = time.time()

while True:
    if test_mode:
        frame = img.copy()
        time.sleep(0.1)  # Simulate video
    else:
        ret, frame = cap.read()
        if not ret:
            break
    
    frame_count += 1
    
    # Resize for speed
    h, w = frame.shape[:2]
    if w > 640:
        frame = cv2.resize(frame, (640, int(h * 640 / w)))
    
    # Run detection
    start_time = time.time()
    detections = detect_objects(frame)
    detection_time = time.time() - start_time
    
    # Draw results
    output_frame = draw_detections(frame, detections)
    
    # Calculate FPS
    if frame_count % 10 == 0:
        fps = 10 / (time.time() - fps_time)
        fps_time = time.time()
    
    # Display info
    cv2.putText(output_frame, f"Office Detector", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(output_frame, f"Objects: {len(detections)}", (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(output_frame, f"FPS: {fps:.1f}", (10, 90),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(output_frame, f"Time: {detection_time*1000:.0f}ms", (10, 120),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    if office_model:
        cv2.putText(output_frame, "Custom Model: ACTIVE", (10, 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    # Show
    cv2.imshow('OFFICE DETECTOR', output_frame)
    
    # Controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        cv2.imwrite(f'office_detection_{int(time.time())}.jpg', output_frame)
        print("💾 Image saved!")
    elif key == ord(' '):
        # Pause
        cv2.waitKey(0)

# Cleanup
if not test_mode:
    cap.release()
cv2.destroyAllWindows()

print("\n" + "="*60)
print(f"📊 DETECTION SUMMARY:")
print(f"   Total frames: {frame_count}")
print(f"   Average FPS: {fps:.1f}")
print(f"   Models: YOLOv8n {'+ Office Model' if office_model else ''}")
print("="*60)
print("✅ DONE!")