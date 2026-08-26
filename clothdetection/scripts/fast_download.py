import urllib.request
import ssl
import os
import json
import random
from pathlib import Path
from io import BytesIO
from PIL import Image

ssl._create_default_https_context = ssl._create_unverified_context

IMAGES = {
    "tops": [
        "https://images.unsplash.com/photo-1576566588028-4147f3842f27?w=400",
        "https://images.unsplash.com/photo-1596755094514-f87e34085b2c?w=400",
        "https://images.unsplash.com/photo-1589310243389-96a5483213a8?w=400",
        "https://images.unsplash.com/photo-1521572163474-6864f9cf17ab?w=400",
        "https://images.unsplash.com/photo-1556909114-f6e7ad7d3136?w=400",
        "https://images.unsplash.com/photo-1608236415058-3c0c018f0ea8?w=400",
        "https://images.unsplash.com/photo-1614252235316-8c857f38b5f4?w=400",
        "https://images.unsplash.com/photo-1602293589930-45aad59ba9c1?w=400",
    ],
    "bottoms": [
        "https://images.unsplash.com/photo-1541099649105-f69ad21f3246?w=400",
        "https://images.unsplash.com/photo-1624378439575-d8705ad7ae80?w=400",
        "https://images.unsplash.com/photo-1593030761757-71fae45fa0e7?w=400",
        "https://images.unsplash.com/photo-1582552938357-32b906df40cb?w=400",
        "https://images.unsplash.com/photo-1604173583682-8f6c70c27a0a?w=400",
        "https://images.unsplash.com/photo-1602293589930-45aad59ba9c1?w=400",
    ],
    "full_body": [
        "https://images.unsplash.com/photo-1595777457583-95e059d581b8?w=400",
        "https://images.unsplash.com/photo-1572804013309-59a88b7e92f1?w=400",
        "https://images.unsplash.com/photo-1593030761757-71fae45fa0e7?w=400",
        "https://images.unsplash.com/photo-1539008835657-9e8e9680c956?w=400",
    ],
    "footwear": [
        "https://images.unsplash.com/photo-1549298916-b41d501d3772?w=400",
        "https://images.unsplash.com/photo-1608256246200-53e635b5b65f?w=400",
        "https://images.unsplash.com/photo-1606107557195-0e29a4b5b4aa?w=400",
        "https://images.unsplash.com/photo-1608236415058-3c0c018f0ea8?w=400",
        "https://images.unsplash.com/photo-1560343090-f0409e92791a?w=400",
    ],
    "accessories": [
        "https://images.unsplash.com/photo-1572635196237-14b3f281503f?w=400",
        "https://images.unsplash.com/photo-1491637639811-60e2756cc1c7?w=400",
        "https://images.unsplash.com/photo-1585386959984-a4155224a1ad?w=400",
        "https://images.unsplash.com/photo-1608236415058-3c0c018f0ea8?w=400",
        "https://images.unsplash.com/photo-1547949003-9792a18a2601?w=400",
    ],
}

base = Path("data/raw")
base.mkdir(parents=True, exist_ok=True)

downloaded = 0
failed = 0

for cat, urls in IMAGES.items():
    cat_dir = base / cat
    cat_dir.mkdir(parents=True, exist_ok=True)
    for i, url in enumerate(urls):
        fname = cat_dir / f"{cat}_{i+1}.jpg"
        if fname.exists():
            downloaded += 1
            continue
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            data = urllib.request.urlopen(req, timeout=15).read()
            img = Image.open(BytesIO(data)).convert("RGB")
            img.save(str(fname), "JPEG", quality=90)
            downloaded += 1
            print(f"  OK {cat}/{cat}_{i+1}.jpg")
        except Exception as e:
            failed += 1
            print(f"  FAIL {cat}_{i+1}: {e}")

print(f"\nDownloaded: {downloaded}, Failed: {failed}")
