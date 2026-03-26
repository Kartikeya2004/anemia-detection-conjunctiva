import cv2
import numpy as np
from ultralytics import YOLO
import tensorflow as tf
from features import extract_color_features

# -------------------------------------------------------
# FIX 1: apply_clahe defined FIRST
# -------------------------------------------------------
def apply_clahe(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# -------------------------------------------------------
# FIX 2: focal_loss needed to load model correctly
# -------------------------------------------------------
def focal_loss(gamma=2., alpha=0.25):
    def loss(y_true, y_pred):
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        pt = tf.exp(-bce)
        return alpha * (1 - pt) ** gamma * bce
    return loss


# Load models once at startup
yolo_model = YOLO("runs/detect/train3/weights/best.pt")
classifier  = tf.keras.models.load_model(
    "anemia_mobilenetv2.keras",
    custom_objects={"loss": focal_loss()}
)

IMAGE_SIZE = 160
THRESHOLD  = 0.55   # FIX 3: optimized threshold


# -------------------------------------------------------
# Grad-CAM
# FIX 4: dual-input model needs both inputs passed
# -------------------------------------------------------
# Replace get_gradcam with this safe version
def get_gradcam(model, img_input, color_features):
    try:
        mobilenet  = model.get_layer("mobilenetv2_1.00_160")
        last_conv  = mobilenet.get_layer("Conv_1")

        # Build a model from image input only through MobileNetV2
        img_only_model = tf.keras.models.Model(
            inputs=mobilenet.input,
            outputs=[last_conv.output, mobilenet.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, _ = img_only_model(img_input)
            # Get full model prediction for loss
            preds = model([img_input, color_features])
            loss  = preds[:, 0]

        grads        = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap      = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap      = tf.squeeze(heatmap)
        heatmap      = tf.maximum(heatmap, 0)
        max_val      = tf.reduce_max(heatmap)
        if max_val > 0:
            heatmap = heatmap / max_val
        return heatmap.numpy()

    except Exception as e:
        print(f"Grad-CAM failed: {e}")
        # Return blank heatmap — app still works
        return np.zeros((10, 10), dtype=np.float32)

# -------------------------------------------------------
# Main prediction function
# -------------------------------------------------------
def predict_image(path):
    img = cv2.imread(path)
    if img is None:
        return "Could not load image", 0.0, path, path

    display = img.copy()

    # YOLO detection
    results  = yolo_model(img, conf=0.20)[0]

    if results.boxes is None or len(results.boxes) == 0:
        return "No conjunctiva detected", 0.0, path, path

    boxes    = results.boxes
    best_idx = int(boxes.conf.argmax())
    x1, y1, x2, y2 = boxes.xyxy[best_idx].cpu().numpy().astype(int)

    h, w   = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return "Invalid crop region", 0.0, path, path

    # Preprocess crop
    crop         = img[y1:y2, x1:x2]
    crop         = apply_clahe(crop)
    crop_resized = cv2.resize(crop, (IMAGE_SIZE, IMAGE_SIZE))
    crop_rgb     = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
    crop_norm    = crop_rgb / 255.0
    crop_input   = np.expand_dims(crop_norm, axis=0).astype(np.float32)

    # FIX 4: extract color features — dual-input model needs both
    color_features = extract_color_features(crop_input)  # shape (1, 4)

    # Predict with dual inputs
    prob = float(classifier.predict(
        [crop_input, color_features], verbose=0
    )[0][0])

    if prob >= THRESHOLD:
        result     = "ANEMIC"
        confidence = prob
        color      = (0, 0, 255)
    else:
        result     = "NON-ANEMIC"
        confidence = 1.0 - prob
        color      = (0, 255, 0)

    # Draw bounding box + label
    cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        display, f"{result} ({confidence:.2f})",
        (x1, max(y1 - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
    )

    # Grad-CAM
    heatmap = get_gradcam(classifier, crop_input, color_features)
    heatmap = cv2.resize(heatmap, (x2 - x1, y2 - y1))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlay = display.copy()
    overlay[y1:y2, x1:x2] = cv2.addWeighted(
        display[y1:y2, x1:x2], 0.6, heatmap, 0.4, 0
    )

    # Save outputs
    import os
    os.makedirs("static", exist_ok=True)
    result_path  = "static/result.jpg"
    gradcam_path = "static/gradcam.jpg"

    cv2.imwrite(result_path, display)
    cv2.imwrite(gradcam_path, overlay)

    return result, confidence, result_path, gradcam_path