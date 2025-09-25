from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd
from threading import Thread
import uuid

from train_pipeline import clean_dataset, merge_datasets, train_model

new_model = None
jobs = {}  # jobId -> status/results

# Load the base model (static, pre-trained)
with open("final_model.pkl", "rb") as f:
    base_model = pickle.load(f)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route("/", methods=["GET"])
def home():
    return {"message": "Welcome to Exo-AI API!"}


@app.route("/predict", methods=["POST"])
def predict_base():
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


# ---------- TRAINING (ASYNC) ----------
def background_train(job_id, files, form_data):
    global new_model
    try:
        # Load CSVs
        stellar = clean_dataset(pd.read_csv(files["stellar"]))
        fpp = clean_dataset(pd.read_csv(files["fpp"]))
        tce = clean_dataset(pd.read_csv(files["tce"]))
        koi = clean_dataset(pd.read_csv(files["koi"]))

        # Params
        n_estimators = int(form_data.get("n_estimators", 500))
        random_state = int(form_data.get("random_state", 42))
        test_size = float(form_data.get("test_size", 0.2))

        # Merge & clean
        master = merge_datasets(stellar, fpp, tce, koi)
        master = clean_dataset(master)

        # Train
        model, features, metrics = train_model(
            master,
            n_estimators=n_estimators,
            random_state=random_state,
            test_size=test_size
        )

        # Store model
        new_model = model

        # Make JSON safe
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

        jobs[job_id] = {
            "status": "done",
            "features": safe_features,
            "metrics": safe_metrics
        }
    except Exception as e:
        jobs[job_id] = {"status": "error", "error": str(e)}


@app.route("/train", methods=["POST"])
def train_new_model():
    try:
        required_keys = ["stellar",  "fpp", "tce", "koi"]
        files = request.files

        # Check required files
        if not all(k in files for k in required_keys):
            return jsonify({"error": f"Missing required files: {required_keys}"}), 400

        # Copy files into memory so the thread can use them later
        file_data = {k: files[k].read() for k in required_keys}
        form_data = request.form.to_dict()

        # Create job
        job_id = str(uuid.uuid4())
        jobs[job_id] = {"status": "running"}

        # Start background thread
        def run_background():
            from io import BytesIO
            try:
                # Recreate file-like objects from memory
                file_objs = {k: BytesIO(v) for k, v in file_data.items()}
                background_train(job_id, file_objs, form_data)
            except Exception as e:
                jobs[job_id] = {"status": "error", "error": str(e)}

        thread = Thread(target=run_background)
        thread.start()

        # Immediate response
        return jsonify({"message": "Training started", "jobId": job_id})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/train-status/<job_id>", methods=["GET"])
def train_status(job_id):
    return jsonify(jobs.get(job_id, {"status": "not found"}))

@app.route("/has-trained-model", methods=["GET"])
def has_trained_model():
    global new_model
    if new_model is not None:
        return jsonify({"has_trained_model": True})
    else:
        return jsonify({"has_trained_model": False})


# ---------- PREDICT USING NEW MODEL ----------
@app.route("/predict-new-model", methods=["POST"])
def predict_new_model():
    global new_model
    try:
        if new_model is None:
            return jsonify({"error": "No trained new model found. Train first at /train"}), 400

        data = request.get_json()
        features = data.get("features")
        if not features:
            return jsonify({"error": "No features provided"}), 400

        features_array = np.array(features).reshape(1, -1)
        prediction = new_model.predict(features_array)
        return jsonify({"prediction": int(prediction[0])})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(debug=True, host="0.0.0.0", port=port)
