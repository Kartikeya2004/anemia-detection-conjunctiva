# Run using:
# python train_model.py 2>$null

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import accuracy_score

# -------------------------------
# CLAHE FUNCTION (TRAIN ONLY)
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
# SETTINGS
# -------------------------------
IMAGE_SIZE = 160
BATCH_SIZE = 16
EPOCHS_STAGE1 = 8
EPOCHS_STAGE2 = 15


# -------------------------------
# DATA GENERATORS
# -------------------------------

# Training generator
train_datagen = ImageDataGenerator(
    preprocessing_function=apply_clahe,
    validation_split=0.2,
    rotation_range=5,
    zoom_range=0.05,
    brightness_range=[0.8, 1.2],
    horizontal_flip=False
)

# Validation generator (NO augmentation, only normalization)
val_datagen = ImageDataGenerator(
    rescale=1./255,
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
# MODEL
# -------------------------------
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)
)

# -------- STAGE 1: Train head only --------
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(64, activation="relu")(x)
output = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
)

print("\n--- Stage 1: Training classification head ---")
model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS_STAGE1,
    class_weight={0: 1.0, 1: 1.2}
)


# -------- STAGE 2: Fine-tune last 50 layers --------
print("\n--- Stage 2: Fine-tuning backbone ---")

for layer in base_model.layers[:-50]:
    layer.trainable = False

for layer in base_model.layers[-50:]:
    layer.trainable = True

model.compile(
    optimizer=Adam(learning_rate=5e-6),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

lr_reduce = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.3,
    patience=2,
    min_lr=1e-6
)

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS_STAGE2,
    class_weight={0: 1.0, 1: 1.2},
    callbacks=[early_stop, lr_reduce]
)


# -------------------------------
# THRESHOLD OPTIMIZATION
# -------------------------------
print("\n🔍 Threshold Optimization Results:")

val_probs = model.predict(val_data)
val_labels = val_data.classes

best_acc = 0
best_t = 0.5

for t in np.arange(0.30, 0.71, 0.05):
    preds = (val_probs > t).astype(int)
    acc = accuracy_score(val_labels, preds)
    print(f"Threshold {t:.2f} -> Accuracy: {acc:.4f}")
    
    if acc > best_acc:
        best_acc = acc
        best_t = t

print(f"\n✅ Best Threshold: {best_t:.2f}")
print(f"✅ Best Validation Accuracy: {best_acc:.4f}")


# -------------------------------
# SAVE MODEL
# -------------------------------
model.save("anemia_mobilenetv2.keras")
print("\nModel saved successfully.")
