# Run using:
# python train_model.py 2>$null
import matplotlib.pyplot as plt
import os
import seaborn as sns
import time
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix
import random

# NEW: import features.py
from features import extract_color_features

tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)

BEST_MODEL_PATH = "best_model.keras"
BEST_SCORE_PATH = "best_score.txt"


# -------------------------------
# CLAHE
# -------------------------------
def apply_clahe(img):
    img = np.uint8(img * 255)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return img / 255.0


# -------------------------------
# FOCAL LOSS
# -------------------------------
def focal_loss(gamma=2., alpha=0.25):
    def loss(y_true, y_pred):
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        pt = tf.exp(-bce)
        return alpha * (1 - pt) ** gamma * bce
    return loss


# -------------------------------
# SETTINGS
# -------------------------------
IMAGE_SIZE = 160
BATCH_SIZE = 16
EPOCHS_STAGE1 = 8
EPOCHS_STAGE2 = 15


# -------------------------------
# DATA GENERATORS
# -------------------------------
train_datagen = ImageDataGenerator(
    preprocessing_function=apply_clahe,
    validation_split=0.2,
    rotation_range=10,
    zoom_range=0.1,
    brightness_range=[0.8, 1.2],
    horizontal_flip=True,
    channel_shift_range=20,
    shear_range=0.1
)

val_datagen = ImageDataGenerator(
    preprocessing_function=apply_clahe,
    validation_split=0.2
)

train_data = train_datagen.flow_from_directory(
    "dataset",
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training",
    shuffle=True
)

val_data = val_datagen.flow_from_directory(
    "dataset",
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation",
    shuffle=False
)


# -------------------------------
# EXTRACT COLOR FEATURES
# NEW: load all val images to extract color features for evaluation
# -------------------------------
def load_all_images(generator):
    images, labels = [], []
    generator.reset()
    for i in range(len(generator)):
        X_batch, y_batch = generator[i]
        images.append(X_batch)
        labels.append(y_batch)
    return np.vstack(images), np.concatenate(labels)

print("Extracting color features from validation set...")
val_images, val_labels_full = load_all_images(val_data)
val_color_features = extract_color_features(val_images)
print(f"Color features shape: {val_color_features.shape}")  # (N, 4)


# -------------------------------
# MODEL — dual input
# NEW: CNN branch + color feature branch merged together
# -------------------------------
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)
)

# --- Image input branch (CNN) ---
image_input = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), name="image_input")
base_model.trainable = False
cnn_out = base_model(image_input)
cnn_out = GlobalAveragePooling2D()(cnn_out)
cnn_out = Dropout(0.2)(cnn_out)

# --- Color feature branch ---
# 4 features: mean_R, mean_G, mean_B, redness_ratio
color_input = Input(shape=(4,), name="color_input")
color_out   = Dense(16, activation="relu")(color_input)

# --- Merge both branches ---
merged = Concatenate()([cnn_out, color_out])
x      = Dense(128, activation="relu")(merged)
x      = Dropout(0.3)(x)
output = Dense(1, activation="sigmoid")(x)

model = Model(inputs=[image_input, color_input], outputs=output)

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss=focal_loss(),
    metrics=[
        "accuracy",
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall")
    ]
)


# -------------------------------
# CUSTOM TRAINING LOOP
# NEW: needed because we have dual inputs (image + color features)
# Standard model.fit with ImageDataGenerator only supports single input
# -------------------------------
def train_epoch(generator, color_feat_all, class_weight):
    generator.reset()
    losses = []
    for i in range(len(generator)):
        X_batch, y_batch = generator[i]
        # Get color features for this batch
        start = i * generator.batch_size
        end   = start + len(X_batch)
        color_batch = color_feat_all[start:end]

        loss = model.train_on_batch(
            [X_batch, color_batch], y_batch,
            class_weight=class_weight
        )
        losses.append(loss[0])
    return np.mean(losses)


def eval_epoch(images, color_features, labels):
    probs = model.predict([images, color_features], verbose=0)
    preds = (probs > 0.5).astype(int).flatten()
    labels = labels.astype(int)
    acc = np.mean(preds == labels)
    return acc


# Extract color features for training set too
print("Extracting color features from training set...")
train_images, train_labels_full = load_all_images(train_data)
train_color_features = extract_color_features(train_images)

# -------- STAGE 1 --------
print("\n--- Stage 1: Training classification head ---")
class_weight = {0: 1.0, 1: 1.3}

for epoch in range(EPOCHS_STAGE1):
    loss = train_epoch(train_data, train_color_features, class_weight)
    val_acc = eval_epoch(val_images, val_color_features, val_labels_full)
    print(f"Epoch {epoch+1}/{EPOCHS_STAGE1} — loss: {loss:.4f} — val_accuracy: {val_acc:.4f}")


# -------- STAGE 2 --------
print("\n--- Stage 2: Fine-tuning backbone ---")

for layer in base_model.layers[:-50]:
    layer.trainable = False
for layer in base_model.layers[-50:]:
    layer.trainable = True

model.compile(
    optimizer=Adam(learning_rate=5e-6),
    loss=focal_loss(),
    metrics=[
        "accuracy",
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
    ]
)

best_val_auc  = 0
patience_count = 0
PATIENCE = 7

for epoch in range(EPOCHS_STAGE2):
    loss = train_epoch(train_data, train_color_features, class_weight)

    # Evaluate AUC on val set
    val_probs = model.predict([val_images, val_color_features], verbose=0)
    from sklearn.metrics import roc_auc_score
    val_auc = roc_auc_score(val_labels_full.astype(int), val_probs)

    print(f"Epoch {epoch+1}/{EPOCHS_STAGE2} — loss: {loss:.4f} — val_auc: {val_auc:.4f}")

    # Manual early stopping on val_auc
    if val_auc > best_val_auc:
        best_val_auc   = val_auc
        patience_count = 0
        model.save("anemia_mobilenetv2.keras")  # save best so far
        print(f"  val_auc improved → {best_val_auc:.4f} — model saved")
    else:
        patience_count += 1
        print(f"  no improvement ({patience_count}/{PATIENCE})")
        if patience_count >= PATIENCE:
            print("Early stopping triggered")
            break

# Reload best weights
model = tf.keras.models.load_model(
    "anemia_mobilenetv2.keras",
    custom_objects={"loss": focal_loss()}
)


# -------------------------------
# THRESHOLD OPTIMIZATION
# -------------------------------
print("\n🔍 Threshold Optimization Results:")

val_probs  = model.predict([val_images, val_color_features], verbose=0)
val_labels = val_labels_full.astype(int)

best_t     = 0.55
best_score = 0

for t in np.arange(0.30, 0.71, 0.05):
    preds       = (val_probs > t).astype(int).flatten()
    tn, fp, fn, tp = confusion_matrix(val_labels, preds).ravel()

    recall      = tp / (tp + fn + 1e-7)
    specificity = tn / (tn + fp + 1e-7)

    print(f"Threshold {t:.2f} -> Recall: {recall:.3f}, Specificity: {specificity:.3f}")

    score = (1.2 * recall) + specificity
    if score > best_score:
        best_score = score
        best_t     = t

print(f"\n✅ Best Threshold: {best_t:.2f}")
print(f"✅ Best Score: {best_score:.4f}")


# -------------------------------
# CONFUSION MATRIX
# -------------------------------
final_preds = (val_probs > best_t).astype(int).flatten()
cm = confusion_matrix(val_labels, final_preds)

print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("confusion_matrix_latest.png")
plt.savefig(f"confusion_matrix_{int(time.time())}.png")
plt.close()

tn, fp, fn, tp = cm.ravel()

sensitivity = tp / (tp + fn + 1e-7)
specificity = tn / (tn + fp + 1e-7)
precision   = tp / (tp + fp + 1e-7)

print("\nMedical Evaluation Metrics:")
print(f"Sensitivity (Recall for Anemia): {sensitivity:.3f}")
print(f"Specificity:                     {specificity:.3f}")
print(f"Precision:                       {precision:.3f}")


# -------------------------------
# SAVE BEST MODEL
# -------------------------------
current_score = sensitivity + specificity
print(f"\nModel Score (Sensitivity + Specificity): {current_score:.4f}")

if os.path.exists(BEST_SCORE_PATH):
    with open(BEST_SCORE_PATH, "r") as f:
        best_saved = float(f.read())
else:
    best_saved = 0.0

if current_score >= best_saved and sensitivity >= 0.75 and specificity >= 0.55:
    print("🚀 New BEST model found! Saving...")
    model.save(BEST_MODEL_PATH)
    with open(BEST_SCORE_PATH, "w") as f:
        f.write(str(current_score))
    print(f"✅ Best model saved with score: {current_score:.4f}")
else:
    print(f"❌ Model not better than best ({best_saved:.4f})")

model.save("anemia_mobilenetv2.keras")
print("\nLatest model saved.")