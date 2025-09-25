import os
from PIL import Image
import pytesseract

INPUT_DIR = "images"
OUTPUT_DIR = "output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

for filename in os.listdir(INPUT_DIR):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
        image_path = os.path.join(INPUT_DIR, filename)
        text = pytesseract.image_to_string(Image.open(image_path), lang='eng')
        base_name = os.path.splitext(filename)[0]
        output_path = os.path.join(OUTPUT_DIR, f"{base_name}.txt")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"OCR complete: {filename} -> {output_path}")