from pathlib import Path
from PIL import Image
import io

def save_image(image_bytes: bytes, output_path: str):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(image_bytes)

def load_image(image_path: str) -> Image.Image:
    return Image.open(image_path)

def resize_image(image: Image.Image, width: int, height: int) -> Image.Image:
    return image.resize((width, height))
