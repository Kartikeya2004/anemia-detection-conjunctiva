from flask import Flask, render_template, request
import os
from predict import predict_image

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return "No file uploaded"

    file = request.files["file"]

    if file.filename == "":
        return "No selected file"

    os.makedirs("static", exist_ok=True)

    filepath = os.path.join("static", file.filename)
    file.save(filepath)

    result, confidence, detected, gradcam = predict_image(filepath)

    return render_template(
        "index.html",
        prediction=result,
        confidence=round(confidence * 100, 2),
        detected=detected,
        gradcam=gradcam,
        original=filepath
    )

if __name__ == "__main__":
    app.run(debug=True)