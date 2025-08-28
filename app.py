# app.py

from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import os

# Load the trained model
model_path = 'trained_model'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)
CORS(app)  # allow requests from frontend

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Placement Prediction API is running"}), 200

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        features = data.get("features", None)

        if not features:
            return jsonify({"error": "Missing 'features' in request body"}), 400

        final_features = np.array([features])
        prediction = model.predict(final_features)
        output = "Placed" if prediction[0] == 1 else "Not Placed"

        return jsonify({
            "prediction": int(prediction[0]),
            "label": output
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
