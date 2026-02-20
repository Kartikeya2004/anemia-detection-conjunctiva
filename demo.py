# use    python demo.py 2>$null   to ignore libpng warning because of corrupted metadata



print("▶ Demo started")

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression

from preprocess import X, y
from features import extract_color_features

print("✅ Dataset loaded")

# -----------------------------
# Train model
# -----------------------------
X_features = extract_color_features(X)
print("✅ Features extracted")

model = LogisticRegression(max_iter=1000)
model.fit(X_features, y)
print("✅ Model trained")

# -----------------------------
# Load test image
# -----------------------------
IMAGE_SIZE = 128
img_path = "test_image.jpg"

img = cv2.imread(img_path)

if img is None:
    print("❌ ERROR: test_image.jpg not found or unreadable")
    exit()

print("✅ Test image loaded")

img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img / 255.0

# -----------------------------
# Extract features
# -----------------------------
test_features = extract_color_features(np.array([img]))

# -----------------------------
# Predict
# -----------------------------
prediction = model.predict(test_features)[0]
confidence = model.predict_proba(test_features)[0]

print("\n🔍 LIVE DEMO RESULT")

if prediction == 1:
    print("🩸 Prediction: ANEMIC")
    print(f"Confidence: {confidence[1]*100:.2f}%")
else:
    print("✅ Prediction: NON-ANEMIC")
    print(f"Confidence: {confidence[0]*100:.2f}%")
