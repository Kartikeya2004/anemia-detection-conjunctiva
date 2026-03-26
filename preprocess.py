from PIL import Image
import os
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

import cv2
import numpy as np

# Base directory
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset")

# FIX 1: was 128 — model expects 160x160
IMAGE_SIZE = 160

# Data containers
X = []
y = []

# Class labels
labels = {
    "anemic": 1,
    "normal": 0
}

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")


# -------------------------------------------------------
# FIX 2: apply_clahe replaces cv2.convertScaleAbs
# Old: simple contrast stretch — different from training
# New: CLAHE on LAB space — matches train_model.py exactly
# -------------------------------------------------------
def apply_clahe(img_bgr):
    img_uint8 = np.uint8(img_bgr)
    lab = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    img_rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return img_rgb / 255.0


# Loop through each class folder
for folder, label in labels.items():
    class_path = os.path.join(DATASET_PATH, folder)

    if not os.path.exists(class_path):
        print(f"Warning: folder not found — {class_path}")
        continue

    for root, dirs, files in os.walk(class_path):
        for img_name in files:

            if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            img_path = os.path.join(root, img_name)

            # Load using OpenCV
            img = cv2.imread(img_path)

            # Fallback to PIL if OpenCV fails
            if img is None:
                try:
                    pil_img = Image.open(img_path).convert("RGB")
                    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                except Exception:
                    print(f"Could not load: {img_path}")
                    continue

            # Handle grayscale
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            # FIX 2: CLAHE preprocessing (replaces convertScaleAbs)
            img = apply_clahe(img)           # returns float32 RGB in [0,1]

            # FIX 1: resize to 160x160
            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))

            X.append(img)
            y.append(label)

# Convert to NumPy arrays
X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int32)

# Print summary
print("Images shape:", X.shape)    # should be (N, 160, 160, 3)
print("Labels shape:", y.shape)
print(f"Anemic:  {np.sum(y == 1)}")
print(f"Normal:  {np.sum(y == 0)}")