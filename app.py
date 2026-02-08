from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load the previously trained logistic regression model, scaler, and label encoder
loaded_model = joblib.load('logistic_regression_model.joblib')
loaded_scaler = joblib.load('scaler.joblib')
loaded_label_encoder = joblib.load('label_encoder.joblib')

print("All objects loaded successfully:")
print("Type of loaded_model:", type(loaded_model))
print("Type of loaded_scaler:", type(loaded_scaler))
print("Type of loaded_label_encoder:", type(loaded_label_encoder))

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if request.is_json:
        try:
            data = request.get_json()
            # Convert input JSON to a DataFrame
            input_df = pd.DataFrame([data])

            # Scale input features using the loaded scaler
            input_scaled = loaded_scaler.transform(input_df)

            # Make predictions using the loaded model
            prediction_encoded = loaded_model.predict(input_scaled)

            # Decode the prediction back to original label
            prediction_label = loaded_label_encoder.inverse_transform(prediction_encoded)

            return jsonify({"prediction": prediction_label[0]})
        except Exception as e:
            return jsonify({"error": f"Error during prediction: {str(e)}"}), 400
    else:
        return jsonify({"error": "Request must be JSON"}), 400

if __name__ == "__main__":
    print("Starting prediction API...")
    app.run(debug=True)
