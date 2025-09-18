from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd
from train_pipeline import clean_dataset, merge_datasets, train_model

new_model = None
# Load the base model (static, pre-trained)
with open("final_model.pkl", "rb") as f:
    base_model = pickle.load(f)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "https://exoai1.vercel.app"}})

@app.route("/", methods=["GET"])
def home():
    return {"message": "Welcome to Exo-AI API!"}


@app.route("/predict", methods=["POST"])
def predict_base():
    """
    Predict using the base static model
    """
    try:
        data = request.get_json()
        features = data.get("features")
        if not features:
            return jsonify({"error": "No features provided"}), 400

        features_array = np.array(features).reshape(1, -1)
        prediction = base_model.predict(features_array)
        return jsonify({"prediction": int(prediction[0])})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/train", methods=["POST"])
def train_new_model():
    """
    Train a new RandomForest model dynamically.
    Expects CSV files sent via multipart/form-data with the keys:
    stellar, toi, fpp, tce, koi
    Optionally accepts n_estimators, random_state, test_size as form fields.
    """
    try:
        required_keys = ["stellar", "toi", "fpp", "tce", "koi"]
        files = request.files

        # Check all required files exist
        if not all(k in files for k in required_keys):
            return jsonify({"error": f"Missing required files: {required_keys}"}), 400

        # Load CSVs into DataFrames and clean
        stellar = clean_dataset(pd.read_csv(files["stellar"]))
        toi = clean_dataset(pd.read_csv(files["toi"]))
        fpp = clean_dataset(pd.read_csv(files["fpp"]))
        tce = clean_dataset(pd.read_csv(files["tce"]))
        koi = clean_dataset(pd.read_csv(files["koi"]))

        # Get optional training parameters from form
        n_estimators = request.form.get("n_estimators", default=500, type=int)
        random_state = request.form.get("random_state", default=42, type=int)
        test_size = request.form.get("test_size", default=0.2, type=float)

        # Merge and clean
        master = merge_datasets(stellar, toi, fpp, tce, koi)
        master = clean_dataset(master)

        # Train model with user-supplied parameters
        model, features, metrics = train_model(
            master,
            n_estimators=n_estimators,
            random_state=random_state,
            test_size=test_size
        )

        # Save model temporarily (can replace with per-user saving later)
        new_model = model

        # --- JSON safe conversion ---
        def make_json_safe(obj):
            if isinstance(obj, (np.ndarray, list)):
                return [make_json_safe(x) for x in obj]
            elif isinstance(obj, (np.int64, np.int32, np.int16)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32, np.float16)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: make_json_safe(v) for k, v in obj.items()}
            return obj

        safe_features = make_json_safe(features)
        safe_metrics = make_json_safe(metrics)

        return jsonify({
            "message": "Model trained successfully!",
            "features": safe_features,
            "metrics": safe_metrics
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict-new-model", methods=["POST"])
def predict_new_model():
    """
    Predict using the dynamically trained model
    """
    try:
        data = request.get_json()
        features = data.get("features")
        if not features:
            return jsonify({"error": "No features provided"}), 400



        features_array = np.array(features).reshape(1, -1)
        prediction = new_model.predict(features_array)
        return jsonify({"prediction": int(prediction[0])})
    except FileNotFoundError:
        return jsonify({"error": "No trained new model found. Train first at /train"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(debug=True, host="0.0.0.0", port=port)
