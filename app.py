import pandas as pd
from flask import Flask, request, jsonify
import joblib

# Define filenames for the saved components
MODEL_FILENAME = 'logistic_regression_model.joblib'
SCALER_FILENAME = 'scaler.joblib'
LABEL_ENCODER_FILENAME = 'label_encoder.joblib'
FEATURE_COLUMNS_FILENAME = 'feature_columns.joblib'

# Load the trained model, scaler, label encoder, and feature column names
try:
    loaded_model = joblib.load(MODEL_FILENAME)
    loaded_scaler = joblib.load(SCALER_FILENAME)
    loaded_label_encoder = joblib.load(LABEL_ENCODER_FILENAME)
    feature_columns = joblib.load(FEATURE_COLUMNS_FILENAME)
    print("All necessary model components loaded successfully.")
except Exception as e:
    print(f"Error loading model components: {e}")
    # In a real application, you might want to handle this error more robustly
    # For now, we'll assume successful loading or fail early during deployment.
    loaded_model = None
    loaded_scaler = None
    loaded_label_encoder = None
    feature_columns = []

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if loaded_model is None or loaded_scaler is None or loaded_label_encoder is None or not feature_columns:
        return jsonify({"error": "Model components not loaded. Please check server logs."}), 500

    if request.is_json:
        try:
            data = request.get_json()

            # Convert input data to a pandas DataFrame, ensuring column order
            input_df = pd.DataFrame([data], columns=feature_columns)

            # Scale the input features
            input_scaled = loaded_scaler.transform(input_df)

            # Make prediction
            prediction_encoded = loaded_model.predict(input_scaled)

            # Decode the prediction back to original label
            prediction_label = loaded_label_encoder.inverse_transform(prediction_encoded)

            return jsonify({"prediction": prediction_label[0]})
        except Exception as e:
            return jsonify({"error": f"Error during prediction: {str(e)}"}), 400
    else:
        return jsonify({"error": "Request must be JSON"}), 400

if __name__ == '__main__':
    # For local execution, run: python app.py
    # For Colab, further steps are needed to expose it publicly (e.g., ngrok)
    app.run(host='0.0.0.0', port=5000)
