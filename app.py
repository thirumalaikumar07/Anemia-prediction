from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# Load the previously trained logistic regression model, scaler, and label encoder
loaded_model = joblib.load('logistic_regression_model.joblib')
loaded_scaler = joblib.load('scaler.joblib')
loaded_label_encoder = joblib.load('label_encoder.joblib')
REQUIRED_FEATURES = ["gender", "hemoglobin", "mch", "mchc", "mcv"]

@app.route("/")
def home():
    return "Anemia Prediction API is running"
if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    
    try:
        data = request.get_json()
        
        # Validate required features
        if not all(feature in data for feature in REQUIRED_FEATURES):
            return jsonify({"error": f"Missing required features: {REQUIRED_FEATURES}"}), 400
        
        # Convert input JSON to a DataFrame with correct feature order
        input_df = pd.DataFrame([[data[f] for f in REQUIRED_FEATURES]], columns=REQUIRED_FEATURES)
        
        # Scale input features using the loaded scaler
        input_scaled = loaded_scaler.transform(input_df)
        
        # Make predictions using the loaded model
        prediction = loaded_model.predict(input_scaled)[0]
        
        # Decode the prediction back to original label
        prediction_label = loaded_label_encoder.inverse_transform([int(prediction)])[0]
        
        return jsonify({"prediction": prediction_label})
        
    except Exception as e:
        return jsonify({"error": f"Error during prediction: {str(e)}"}), 400

if __name__ == "__main__":
    print("Starting prediction API...")
    app.run(debug=True)
