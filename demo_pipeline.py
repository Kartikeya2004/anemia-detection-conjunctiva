import cv2
import numpy as np
import sys
from ultralytics import YOLO
import tensorflow as tf

print("🔄 Loading models...")

# -----------------------------
# LOAD MODELS
# -----------------------------
yolo_model = YOLO("runs/detect/train3/weights/best.pt")
classifier = tf.keras.models.load_model("anemia_mobilenetv2.keras")

print("✅ Models loaded successfully")

IMAGE_SIZE = 160
THRESHOLD = 0.45   # ✅ optimized threshold

# -----------------------------
# LOAD IMAGE
# -----------------------------
image_path = "demo/test.jpg"   # <-- put demo images here
img = cv2.imread(image_path)

if img is None:
    print("❌ Image not found")
    sys.exit()

# -----------------------------
# YOLO DETECTION
# -----------------------------
results = yolo_model(img, conf=0.20)[0]

if results.boxes is None or len(results.boxes) == 0:
    print("❌ No conjunctiva detected")
    sys.exit()

boxes = results.boxes

# choose highest confidence box
best_idx = boxes.conf.argmax()
box = boxes.xyxy[best_idx].cpu().numpy().astype(int)

x1, y1, x2, y2 = box

# safety clipping
h, w = img.shape[:2]
x1, y1 = max(0, x1), max(0, y1)
x2, y2 = min(w, x2), min(h, y2)

crop = img[y1:y2, x1:x2]

if crop.size == 0:
    print("❌ Invalid crop")
    sys.exit()

# -----------------------------
# PREPROCESS
# -----------------------------
crop = cv2.resize(crop, (IMAGE_SIZE, IMAGE_SIZE))
crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
crop = crop / 255.0
crop = np.expand_dims(crop, axis=0)

# -----------------------------
# CLASSIFICATION
# -----------------------------
prob = classifier.predict(crop, verbose=0)[0][0]
print("Raw model probability:", prob)

if prob > THRESHOLD:
    result = "ANEMIC"
    confidence = prob
    color = (0, 0, 255)
else:
    result = "NON-ANEMIC"
    confidence = 1 - prob
    color = (0, 255, 0)

# -----------------------------
# DISPLAY RESULT
# -----------------------------
cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

text = f"{result} ({confidence:.2f})"
cv2.putText(img, text, (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

cv2.imshow("Anemia Detection Demo", img)

print("\n✅ Press ENTER to close demo")

# ENTER closes window
while True:
    key = cv2.waitKey(1)
    if key == 13:
        break

cv2.destroyAllWindows()
print("👋 Demo finished.")