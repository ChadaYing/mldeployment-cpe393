# Import required libraries
from flask import Flask, request, jsonify  # For creating the web API
import pickle                              # For loading the trained model
import numpy as np                         # For handling numerical input data

# Create a Flask app instance
app = Flask(__name__)

# Load the trained model from the file
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Root endpoint (GET request). Just to check if the server is running.
@app.route("/")
def home():
    return "ML Model is Running"

# Predict endpoint (POST request). Accepts input features and returns predictions.
@app.route("/predict", methods=["POST"])
def predict():
    # Parse the JSON input
    data = request.get_json()

    # -------------------------------
    # Input validation
    # -------------------------------

    # Check if the key "features" exists in the input
    if "features" not in data:
        return jsonify({"error": "Missing 'features' key"}), 400

    # Check if the "features" is a list
    if not isinstance(data["features"], list):
        return jsonify({"error": "'features' must be a list"}), 400

    # Check if each item is a list of 4 values
    if not all(isinstance(row, list) and len(row) == 13 for row in data["features"]):
        return jsonify({"error": "Each input must be a list with 13 values"}), 400

    # -------------------------------
    # Valid input: do prediction
    # -------------------------------

    # Convert input features to a NumPy array
    input_features = np.array(data["features"])

    # Predict class labels
    predictions = model.predict(input_features)

    # Simulated confidence score: use std deviation across all trees in RandomForest
    all_tree_predictions = np.stack([tree.predict(input_features) for tree in model.estimators_], axis=1)
    std_devs = np.std(all_tree_predictions, axis=1)

    # Convert standard deviation into a fake confidence score (lower std = higher confidence)
    max_std = std_devs.max() + 1e-6  # avoid divide-by-zero
    confidences = 1 - (std_devs / max_std)

    # Package result
    results = [
        {
            "prediction": round(float(pred), 2),
            "confidence": round(float(conf), 2)
        }
        for pred, conf in zip(predictions, confidences)
    ]

    # Return predictions as JSON
    return jsonify(results)

# Health check endpoint to confirm the API is live
@app.route("/health")
def health():
    return jsonify({"status": "ok"})

# Run the Flask app on port 9000 (0.0.0.0 = allow all IPs inside Docker)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9000)
