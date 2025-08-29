# app.py

from flask import Flask, request, jsonify
import pickle
import numpy as np
from flask_cors import CORS

# Load the trained model
model_path = "final_model.pkl"
with open(model_path, "rb") as file:
    model = pickle.load(file)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

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

        return jsonify({"prediction": int(prediction[0])})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    import os

    port = int(os.environ.get("PORT", 10000))
    app.run(debug=True, host="0.0.0.0", port=port)
