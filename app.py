import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk
from ultralytics import YOLO
import tensorflow as tf

# -----------------------------
# LOAD MODELS
# -----------------------------
print("Loading models...")

yolo_model = YOLO("runs/detect/train3/weights/best.pt")
classifier = tf.keras.models.load_model("anemia_mobilenetv2.keras")

print("Models loaded")

IMAGE_SIZE = 160
THRESHOLD = 0.45

# -----------------------------
# GUI
# -----------------------------
root = tk.Tk()
root.title("Anemia Detection System")

title = tk.Label(root,
                 text="Anemia Detection from Eye Conjunctiva",
                 font=("Arial", 16, "bold"))
title.pack(pady=10)

img_label = tk.Label(root)
img_label.pack()

result_label = tk.Label(root, font=("Arial", 14))
result_label.pack(pady=10)


# -----------------------------
# PROCESS IMAGE
# -----------------------------
def process_image(path):

    img = cv2.imread(path)
    display = img.copy()

    results = yolo_model(img, conf=0.20)[0]

    if results.boxes is None or len(results.boxes) == 0:
        result_label.config(text="No conjunctiva detected",
                            fg="red")
        return

    boxes = results.boxes
    best_idx = boxes.conf.argmax()

    x1, y1, x2, y2 = boxes.xyxy[best_idx].cpu().numpy().astype(int)

    h, w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    crop = img[y1:y2, x1:x2]

    crop = cv2.resize(crop, (IMAGE_SIZE, IMAGE_SIZE))
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    crop = crop / 255.0
    crop = np.expand_dims(crop, axis=0)

    prob = classifier.predict(crop, verbose=0)[0][0]

    if prob > THRESHOLD:
        result = "ANEMIC"
        confidence = prob
        color = (0, 0, 255)
        text_color = "red"
    else:
        result = "NON-ANEMIC"
        confidence = 1 - prob
        color = (0, 255, 0)
        text_color = "green"

    cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

    display = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
    display = cv2.resize(display, (500, 350))

    img_pil = Image.fromarray(display)
    img_tk = ImageTk.PhotoImage(img_pil)

    img_label.configure(image=img_tk)
    img_label.image = img_tk

    result_label.config(
        text=f"Prediction: {result}\nConfidence: {confidence:.2f}",
        fg=text_color
    )


# -----------------------------
# UPLOAD BUTTON
# -----------------------------
def upload():
    file_path = filedialog.askopenfilename(
        filetypes=[("Images", "*.jpg *.png *.jpeg")]
    )
    if file_path:
        process_image(file_path)


btn = tk.Button(root,
                text="Upload Eye Image",
                command=upload,
                font=("Arial", 12))

btn.pack(pady=10)

root.mainloop()