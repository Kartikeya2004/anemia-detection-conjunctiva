# Anemia Detection using Conjunctiva Image Analysis

This project develops a non-invasive anemia screening system using computer vision and deep learning.

The system analyzes eye conjunctiva images to detect signs of anemia.

## Pipeline

Eye Image  
→ YOLO Conjunctiva Detection  
→ CLAHE Preprocessing  
→ MobileNetV2 Classification  
→ Anemia Prediction

## Features

- Automatic conjunctiva detection using YOLO
- Deep learning anemia classification using MobileNetV2
- CLAHE contrast enhancement
- Medical evaluation metrics (Sensitivity, Specificity, Precision)
- End-to-end demo pipeline
- GUI application for testing

## Tech Stack

- Python
- OpenCV
- TensorFlow / Keras
- Ultralytics YOLO
- Scikit-learn

## Results

- Validation Accuracy: ~75%
- AUC Score: ~0.8