# CELL 1: BASIC SETUP ONLY
# !pip install ultralytics albumentations opencv-python pyyaml -q
# !apt-get update && apt-get install -y libgl1-mesa-glx -q

import os
import cv2
import numpy as np
import shutil
import yaml
import random
from pathlib import Path

print("✅ Basic setup complete")


# CELL 2: LOAD YOUR ROOM-DATA DATASET
print("📂 Checking available datasets...")
# !ls /kaggle/input/

# Find your dataset
dataset_path = Path("/kaggle/input")
for item in dataset_path.iterdir():
    print(f"Found: {item.name}")

# Let me check what's actually there
print("\n🔍 Detailed check:")
# !find /kaggle/input -type f -name "*.png" | head -20


# CELL 3: SETUP DIRECTORY AND COPY IMAGES
import os
import shutil
from pathlib import Path

base_dir = Path("/kaggle/working/office_detection")
base_dir.mkdir(exist_ok=True)

raw_images_dir = base_dir / "raw_images"
raw_images_dir.mkdir(exist_ok=True)

# Your image list (from what we found)
your_images = [
    "stool.png", "steamer.png", "calender.png", "cupboard.png", "batmanlogo.png",
    "prayermat.png", "books2.png", "socket.png", "books.png", "printer1.png",
    "crack.png", "pen1.png", "pen.png", "clothes.png", "pinter.png",
    "chair.png", "laptop.png", "books1.png"
]

print("📥 Copying images to working directory...")
found_count = 0
for img_name in your_images:
    src_path = Path("/kaggle/input/room-data") / img_name
    if src_path.exists():
        shutil.copy2(src_path, raw_images_dir / img_name)
        print(f"  ✅ Copied: {img_name}")
        found_count += 1
    else:
        print(f"  ❌ Missing: {img_name}")

print(f"\n📊 Total images copied: {found_count}")
print("\n📸 Your images in working directory:")
for img in sorted(raw_images_dir.glob("*.png")):
    size_kb = img.stat().st_size / 1024
    print(f"  - {img.name} ({size_kb:.1f} KB)")
    
    
# !pip install albumentations -q

# CELL 4: SYNTHETIC DATA WITHOUT ALBUMENTATIONS
import cv2
import numpy as np
import random
import math

print("🔥 GENERATING SYNTHETIC DATA - NO EXTERNAL LIBRARIES")

# Create output directory
augmented_dir = Path("/kaggle/working/office_detection/augmented")
augmented_dir.mkdir(exist_ok=True)

# OBJECT MAPPING
object_mapping = {
    "laptop.png": 0,
    "batmanlogo.png": 1,
    "prayermat.png": 2,
    "cupboard.png": 3,
    "books.png": 4,
    "books1.png": 4,
    "books2.png": 4,
    "steamer.png": 5,
    "chair.png": 6,
    "clothes.png": 7,
    "pinter.png": 8,
    "printer1.png": 8,
    "crack.png": 9,
    "stool.png": 10,
    "calender.png": 11,
    "pen.png": 12,
    "pen1.png": 12,
    "socket.png": 13,
}

# CLASS NAMES
class_names = [
    "laptop", "batman_sticker", "prayer_mat", "cupboard", "books",
    "steamer", "chair", "clothes", "printer", "crack",
    "stool", "calendar", "pen", "socket"
]

print(f"🎯 DETECTING {len(class_names)} OBJECTS")

# MANUAL AUGMENTATION FUNCTIONS
def augment_image(image, bbox):
    """Apply random augmentations"""
    h, w = image.shape[:2]
    output = image.copy()
    
    # 1. Random flip
    if random.random() > 0.5:
        output = cv2.flip(output, 1)
        # Adjust bbox for horizontal flip
        bbox[0] = 1.0 - bbox[0]  # x_center
    
    # 2. Random brightness
    if random.random() > 0.5:
        brightness = random.randint(-50, 50)
        output = cv2.add(output, brightness)
    
    # 3. Random contrast
    if random.random() > 0.5:
        contrast = random.uniform(0.5, 1.5)
        output = cv2.convertScaleAbs(output, alpha=contrast, beta=0)
    
    # 4. Random rotation (simple)
    if random.random() > 0.7:
        angle = random.randint(-30, 30)
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
        output = cv2.warpAffine(output, M, (w, h))
    
    # 5. Random scale (zoom)
    if random.random() > 0.5:
        scale = random.uniform(0.8, 1.2)
        new_w, new_h = int(w * scale), int(h * scale)
        output = cv2.resize(output, (new_w, new_h))
        # Keep at 640x640
        output = cv2.resize(output, (640, 640))
    
    # 6. Add noise
    if random.random() > 0.7:
        noise = np.random.randint(-20, 20, output.shape, dtype=np.int32)
        output = cv2.add(output, noise.astype(np.uint8))
    
    return output, bbox

# GENERATE 40 VARIATIONS PER IMAGE
variations_per_image = 40
total_created = 0

for img_file, class_id in object_mapping.items():
    img_path = raw_images_dir / img_file
    if not img_path.exists():
        print(f"⚠️ Missing: {img_file}")
        continue
    
    # LOAD IMAGE
    image = cv2.imread(str(img_path))
    if image is None:
        print(f"❌ Failed to load: {img_file}")
        continue
    
    # RESIZE TO STANDARD
    image = cv2.resize(image, (640, 640))
    
    # ORIGINAL BBOX (OBJECT COVERS IMAGE)
    x_center, y_center = 0.5, 0.5
    bbox_width, bbox_height = 0.8, 0.8
    original_bbox = [x_center, y_center, bbox_width, bbox_height]
    
    successful = 0
    for i in range(variations_per_image):
        try:
            # Apply augmentations
            aug_image, aug_bbox = augment_image(image, original_bbox.copy())
            
            # Save image
            aug_name = f"{img_file.replace('.png', '')}_{i:03d}.jpg"
            cv2.imwrite(str(augmented_dir / aug_name), aug_image)
            
            # Save label
            label_name = f"{img_file.replace('.png', '')}_{i:03d}.txt"
            with open(augmented_dir / label_name, 'w') as f:
                f.write(f"{class_id} {aug_bbox[0]} {aug_bbox[1]} {aug_bbox[2]} {aug_bbox[3]}\n")
            
            successful += 1
            total_created += 1
            
        except Exception as e:
            # Save original as fallback
            aug_name = f"{img_file.replace('.png', '')}_fallback_{i:03d}.jpg"
            cv2.imwrite(str(augmented_dir / aug_name), image)
            
            label_name = f"{img_file.replace('.png', '')}_fallback_{i:03d}.txt"
            with open(augmented_dir / label_name, 'w') as f:
                f.write(f"{class_id} {x_center} {y_center} {bbox_width} {bbox_height}\n")
            
            successful += 1
            total_created += 1
    
    print(f"✅ {img_file}: {successful} variations")

print(f"\n🎉 TOTAL SYNTHETIC IMAGES: {total_created}")
print(f"   Dataset size: ~{total_created} images")

# CELL 4: SYNTHETIC DATA WITHOUT ALBUMENTATIONS
import cv2
import numpy as np
import random
import math

print("🔥 GENERATING SYNTHETIC DATA - NO EXTERNAL LIBRARIES")

# Create output directory
augmented_dir = Path("/kaggle/working/office_detection/augmented")
augmented_dir.mkdir(exist_ok=True)

# OBJECT MAPPING
object_mapping = {
    "laptop.png": 0,
    "batmanlogo.png": 1,
    "prayermat.png": 2,
    "cupboard.png": 3,
    "books.png": 4,
    "books1.png": 4,
    "books2.png": 4,
    "steamer.png": 5,
    "chair.png": 6,
    "clothes.png": 7,
    "pinter.png": 8,
    "printer1.png": 8,
    "crack.png": 9,
    "stool.png": 10,
    "calender.png": 11,
    "pen.png": 12,
    "pen1.png": 12,
    "socket.png": 13,
}

# CLASS NAMES
class_names = [
    "laptop", "batman_sticker", "prayer_mat", "cupboard", "books",
    "steamer", "chair", "clothes", "printer", "crack",
    "stool", "calendar", "pen", "socket"
]

print(f"🎯 DETECTING {len(class_names)} OBJECTS")

# MANUAL AUGMENTATION FUNCTIONS
def augment_image(image, bbox):
    """Apply random augmentations"""
    h, w = image.shape[:2]
    output = image.copy()
    
    # 1. Random flip
    if random.random() > 0.5:
        output = cv2.flip(output, 1)
        # Adjust bbox for horizontal flip
        bbox[0] = 1.0 - bbox[0]  # x_center
    
    # 2. Random brightness
    if random.random() > 0.5:
        brightness = random.randint(-50, 50)
        output = cv2.add(output, brightness)
    
    # 3. Random contrast
    if random.random() > 0.5:
        contrast = random.uniform(0.5, 1.5)
        output = cv2.convertScaleAbs(output, alpha=contrast, beta=0)
    
    # 4. Random rotation (simple)
    if random.random() > 0.7:
        angle = random.randint(-30, 30)
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
        output = cv2.warpAffine(output, M, (w, h))
    
    # 5. Random scale (zoom)
    if random.random() > 0.5:
        scale = random.uniform(0.8, 1.2)
        new_w, new_h = int(w * scale), int(h * scale)
        output = cv2.resize(output, (new_w, new_h))
        # Keep at 640x640
        output = cv2.resize(output, (640, 640))
    
    # 6. Add noise
    if random.random() > 0.7:
        noise = np.random.randint(-20, 20, output.shape, dtype=np.int32)
        output = cv2.add(output, noise.astype(np.uint8))
    
    return output, bbox

# GENERATE 40 VARIATIONS PER IMAGE
variations_per_image = 40
total_created = 0

for img_file, class_id in object_mapping.items():
    img_path = raw_images_dir / img_file
    if not img_path.exists():
        print(f"⚠️ Missing: {img_file}")
        continue
    
    # LOAD IMAGE
    image = cv2.imread(str(img_path))
    if image is None:
        print(f"❌ Failed to load: {img_file}")
        continue
    
    # RESIZE TO STANDARD
    image = cv2.resize(image, (640, 640))
    
    # ORIGINAL BBOX (OBJECT COVERS IMAGE)
    x_center, y_center = 0.5, 0.5
    bbox_width, bbox_height = 0.8, 0.8
    original_bbox = [x_center, y_center, bbox_width, bbox_height]
    
    successful = 0
    for i in range(variations_per_image):
        try:
            # Apply augmentations
            aug_image, aug_bbox = augment_image(image, original_bbox.copy())
            
            # Save image
            aug_name = f"{img_file.replace('.png', '')}_{i:03d}.jpg"
            cv2.imwrite(str(augmented_dir / aug_name), aug_image)
            
            # Save label
            label_name = f"{img_file.replace('.png', '')}_{i:03d}.txt"
            with open(augmented_dir / label_name, 'w') as f:
                f.write(f"{class_id} {aug_bbox[0]} {aug_bbox[1]} {aug_bbox[2]} {aug_bbox[3]}\n")
            
            successful += 1
            total_created += 1
            
        except Exception as e:
            # Save original as fallback
            aug_name = f"{img_file.replace('.png', '')}_fallback_{i:03d}.jpg"
            cv2.imwrite(str(augmented_dir / aug_name), image)
            
            label_name = f"{img_file.replace('.png', '')}_fallback_{i:03d}.txt"
            with open(augmented_dir / label_name, 'w') as f:
                f.write(f"{class_id} {x_center} {y_center} {bbox_width} {bbox_height}\n")
            
            successful += 1
            total_created += 1
    
    print(f"✅ {img_file}: {successful} variations")

print(f"\n🎉 TOTAL SYNTHETIC IMAGES: {total_created}")
print(f"   Dataset size: ~{total_created} images")

# CELL 5: CREATE DATASET STRUCTURE
import shutil

dataset_dir = Path("/kaggle/working/office_detection/dataset")

# Create directories
(train_dir := dataset_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
(val_dir := dataset_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
(train_labels_dir := dataset_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
(val_labels_dir := dataset_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)

# Get all generated files
all_images = list(augmented_dir.glob("*.jpg"))
print(f"📊 Total images: {len(all_images)}")

# SIMPLE SPLIT
split_idx = int(0.8 * len(all_images))
train_imgs = all_images[:split_idx]
val_imgs = all_images[split_idx:]

print(f"📚 Training images: {len(train_imgs)}")
print(f"📚 Validation images: {len(val_imgs)}")

# COPY FILES
for img_path in train_imgs:
    shutil.copy2(img_path, train_dir / img_path.name)
    label_path = augmented_dir / f"{img_path.stem}.txt"
    if label_path.exists():
        shutil.copy2(label_path, train_labels_dir / f"{img_path.stem}.txt")

for img_path in val_imgs:
    shutil.copy2(img_path, val_dir / img_path.name)
    label_path = augmented_dir / f"{img_path.stem}.txt"
    if label_path.exists():
        shutil.copy2(label_path, val_labels_dir / f"{img_path.stem}.txt")

print("✅ Dataset created!")

# CELL 6: CREATE YAML
data_yaml_content = f"""# Office Objects Dataset
path: {dataset_dir}
train: images/train
val: images/val

# Number of classes
nc: {len(class_names)}

# Class names
names: {class_names}
"""

yaml_path = dataset_dir / "data.yaml"
with open(yaml_path, 'w') as f:
    f.write(data_yaml_content)

print("📄 YAML created:")
print(data_yaml_content)

# CELL 7: VERIFY
print("🔍 Verifying...")
train_count = len(list(train_dir.glob("*.jpg")))
val_count = len(list(val_dir.glob("*.jpg")))

print(f"✅ Train: {train_count} images")
print(f"✅ Val: {val_count} images")

# Check labels
sample_label = list(train_labels_dir.glob("*.txt"))[0]
with open(sample_label, 'r') as f:
    print(f"📝 Sample label: {f.read().strip()}")

print("\n🚀 READY FOR TRAINING!")