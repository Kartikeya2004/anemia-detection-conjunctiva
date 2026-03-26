# Anemia Detection using Conjunctiva Image Analysis

A non-invasive anemia screening system using computer vision and deep learning. The system analyzes eye conjunctiva images to detect signs of anemia — no blood test required.

---

## Pipeline

```
Eye Image
   ↓
YOLO (conjunctiva detection)
   ↓
CLAHE preprocessing (contrast enhancement)
   ↓
Resize to 160×160
   ↓
MobileNetV2 (CNN branch) + Color Features (R/G/B/redness branch)
   ↓
Merged dual-input classifier
   ↓
Probability → Threshold (0.55) → ANEMIC / NON-ANEMIC
```

---

## Features

- Automatic conjunctiva detection using YOLOv8
- Dual-input deep learning model — CNN features + LAB color features
- CLAHE contrast enhancement (matches clinical imaging conditions)
- Grad-CAM visualization — shows which region influenced the prediction
- Medical evaluation metrics: Sensitivity, Specificity, Precision, AUC
- Flask web app with drag-and-drop image upload
- CLI demo pipeline for quick testing

---

## Results

| Metric | Value |
|---|---|
| AUC | 0.945 |
| Sensitivity (Recall) | 0.853 |
| Specificity | 0.937 |
| Precision | 0.900 |
| Best Threshold | 0.55 |
| Dataset size | 415 images (333 train / 82 val) |

---

## Tech Stack

- Python 3.10
- TensorFlow / Keras — MobileNetV2 transfer learning
- Ultralytics YOLOv8 — conjunctiva detection
- OpenCV — CLAHE preprocessing, image processing
- Scikit-learn — evaluation metrics
- Flask — web application
- NumPy, Matplotlib, Seaborn

---

## Project Structure

```
Mini Project/
├── train_model.py        # Train dual-input model (MobileNetV2 + color features)
├── predict.py            # Full inference pipeline (YOLO → CLAHE → classify → Grad-CAM)
├── app.py                # Flask web application
├── features.py           # Color feature extraction (R/G/B/redness ratio)
├── preprocess.py         # Dataset loader with CLAHE preprocessing
├── demo_pipeline.py      # CLI demo — run on a single image
├── best_model.keras      # Best saved model weights
├── dataset/
│   ├── anemic/           # Anemic eye images
│   └── normal/           # Normal eye images
├── runs/detect/          # YOLO model weights
├── static/               # Flask output images
└── templates/            # Flask HTML templates
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Usage

### Web App
```bash
python app.py
# Open http://127.0.0.1:5000
```

### CLI Demo
```bash
# Place test image at demo/test.jpg
python demo_pipeline.py
```

### Train Model
```bash
python train_model.py 2>$null
```

---

## How it Works

The model uses two inputs simultaneously:

**1. CNN branch** — MobileNetV2 extracts visual features from the conjunctiva image (texture, color patterns, pallor)

**2. Color feature branch** — extracts mean R, G, B values and redness ratio directly from the image. Anemic conjunctiva is measurably less red — this gives the model a direct medical signal.

Both branches are merged and passed through a dense classifier. This dual-input approach improved AUC from ~0.80 to 0.945.

---

## Limitations

- Dataset size (415 images) limits generalization
- YOLO may crop incorrectly under poor lighting
- Threshold tuned on validation set — may need adjustment for new populations
- Not a replacement for clinical Hb testing

---

## Future Work

- Regression model to predict actual Hb value (g/dL)
- YOLO segmentation instead of bounding box
- Multi-region detection (conjunctiva + lips + nails)
- Mobile deployment (Flutter + TFLite)
- Larger, standardized clinical dataset