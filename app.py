from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)
CORS(app)

# Load EfficientNetB0 model
model = tf.keras.models.load_model("model_tb_effnetb0.h5")

UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # === Preprocessing ===
    image = Image.open(filepath).convert("RGB")
    image = image.resize((224, 224))
    image = np.array(image).astype("float32")
    image = np.expand_dims(image, axis=0)
    image = tf.keras.applications.efficientnet.preprocess_input(image)

    # === Prediction ===
    prediction = model.predict(image)[0][0]  # Probabilitas TB

    if prediction > 0.5:
        result = "TB"
        confidence = prediction
    else:
        result = "Normal"
        confidence = 1 - prediction

    return jsonify({
        "result": result,
        "confidence": round(float(confidence * 100), 2)  # dalam persen
    })

if __name__ == "__main__":
    app.run(debug=True)