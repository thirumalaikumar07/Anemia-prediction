from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load model & scaler
model = joblib.load("logistic_regression_model.joblib")
scaler = joblib.load("scaler.joblib")

@app.route("/")
def home():
    return "Anemia Prediction API is running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    # Example input order – adjust based on your training
    features = [
        data["gender"],
        data["hemoglobin"],
        data["mch"],
        data["mchc"],
        data["mcv"]
    ]

    features = np.array(features).reshape(1, -1)
    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)

    result = "Anemia" if prediction[0] == 1 else "No Anemia"

    return jsonify({
        "prediction": result
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
