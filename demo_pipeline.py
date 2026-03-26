import cv2
import numpy as np
import sys
from ultralytics import YOLO
import tensorflow as tf


def focal_loss(gamma=2., alpha=0.25):
    def loss(y_true, y_pred):
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        pt = tf.exp(-bce)
        return alpha * (1 - pt) ** gamma * bce
    return loss


# FIX 1: apply_clahe defined here at the top
# Old code had no CLAHE — model was trained with CLAHE, must apply at inference
def apply_clahe(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


print("Loading models...")

yolo_model = YOLO("runs/detect/train3/weights/best.pt")
classifier = tf.keras.models.load_model(
    "anemia_mobilenetv2.keras",
    custom_objects={"loss": focal_loss()}
)

print("Models loaded successfully")

IMAGE_SIZE = 160
THRESHOLD  = 0.55   # FIX 2: was 0.5, use optimized threshold from training

image_path = "demo/test.jpg"
img = cv2.imread(image_path)

if img is None:
    print("Image not found at:", image_path)
    sys.exit()

results = yolo_model(img, conf=0.20)[0]

if results.boxes is None or len(results.boxes) == 0:
    print("No conjunctiva detected")
    sys.exit()

boxes    = results.boxes
best_idx = int(boxes.conf.argmax())
x1, y1, x2, y2 = boxes.xyxy[best_idx].cpu().numpy().astype(int)

h, w   = img.shape[:2]
x1, y1 = max(0, x1), max(0, y1)
x2, y2 = min(w, x2), min(h, y2)

crop = img[y1:y2, x1:x2]

if crop.size == 0:
    print("Invalid crop")
    sys.exit()

# FIX 1: CLAHE applied before resize — matches training pipeline exactly
crop = apply_clahe(crop)
crop = cv2.resize(crop, (IMAGE_SIZE, IMAGE_SIZE))
crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
crop = crop / 255.0
crop = np.expand_dims(crop, axis=0).astype(np.float32)

prob = float(classifier.predict(crop, verbose=0)[0][0])
print(f"Raw model probability: {prob:.4f}")

if prob >= THRESHOLD:
    result     = "ANEMIC"
    confidence = prob
    color      = (0, 0, 255)
else:
    result     = "NON-ANEMIC"
    confidence = 1.0 - prob
    color      = (0, 255, 0)

cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
text = f"{result} ({confidence:.2f})"
cv2.putText(img, text, (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

cv2.imshow("Anemia Detection Demo", img)

print(f"\nResult    : {result}")
print(f"Confidence: {confidence:.2f}")
print(f"Threshold : {THRESHOLD}")
print("\nPress ENTER to close")

while True:
    key = cv2.waitKey(1)
    if key == 13:
        break

cv2.destroyAllWindows()
print("Demo finished.")