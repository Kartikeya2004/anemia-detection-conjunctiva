import os
from PIL import Image

DATASET_PATH = "dataset"

for root, dirs, files in os.walk(DATASET_PATH):
    for file in files:
        if file.lower().endswith(".png"):
            png_path = os.path.join(root, file)
            jpg_path = png_path.replace(".png", ".jpg")

            try:
                img = Image.open(png_path).convert("RGB")
                img.save(jpg_path, "JPEG")
                os.remove(png_path)  # remove old png
            except:
                pass

print("PNG to JPG conversion completed")
