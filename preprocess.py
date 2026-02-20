from PIL import Image
import os
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

import cv2
import numpy as np

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset")

# Image size
IMAGE_SIZE = 128

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

# Loop through each class folder
for folder, label in labels.items():
    class_path = os.path.join(DATASET_PATH, folder)

    if not os.path.exists(class_path):
        continue

    # 🔁 Walk through ALL subfolders (important)
    for root, dirs, files in os.walk(class_path):
        for img_name in files:

            # Accept only image files
            if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            img_path = os.path.join(root, img_name)

            # Load using OpenCV
            img = cv2.imread(img_path)

            # Fallback to PIL if OpenCV fails
            if img is None:
                try:
                    pil_img = Image.open(img_path).convert("RGB")
                    img = np.array(pil_img)
                except:
                    continue

            # Handle grayscale pallor crops
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            # 🔥 Contrast enhancement for conjunctival pallor
            img = cv2.convertScaleAbs(img, alpha=1.4, beta=20)

            # Resize
            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))

            # Convert to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Normalize (DO THIS LAST)
            img = img / 255.0

            # Append data
            X.append(img)
            y.append(label)

# Convert to NumPy arrays
X = np.array(X)
y = np.array(y)

# Print summary
print("Images shape:", X.shape)
print("Labels shape:", y.shape)
