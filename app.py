# app.py

from flask import Flask, request, jsonify
import pickle
import numpy as np
from flask_cors import CORS
# Load the trained model
model_path = "trained_model.pkl"
with open(model_path, "rb") as file:
    model = pickle.load(file)

app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET"])
def home():
    return {"message": "Welcome to Exo-AI API!"}


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Expecting JSON like: { "features": [0.1, 0.2, ..., 58 values] }
        features = data.get("features")
        if features is None:
            return jsonify({"error": "No features provided"}), 400

        # Convert to numpy array
        features_array = np.array(features).reshape(1, -1)

        # Make prediction
        prediction = model.predict(features_array)
        pred = int(prediction[0])

        # Map prediction to class label
        prediction_labels = {
            0: "FALSE POSITIVE",
            1: "CONFIRMED",
            2: "CANDIDATE",
            3: "NOT DISPOSITIONED"
        }
        label = prediction_labels.get(pred, "Unknown")

        return jsonify({
            "prediction": pred,
            "label": label
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    import os

    port = int(os.environ.get("PORT", 10000))
    app.run(debug=True, host="0.0.0.0", port=port)
