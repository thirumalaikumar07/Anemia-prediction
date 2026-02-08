from flask import Flask, request, jsonify
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Re-initialize the Flask application instance to ensure a clean state
app = Flask(__name__)

# Load the previously trained logistic regression model
loaded_model = joblib.load('logistic_regression_model.joblib')
print("Logistic Regression model loaded successfully.")

# Re-instantiate and fit StandardScaler to the original X data (assuming X is globally available)
# In a real API deployment, X would not be available; the scaler object itself would be saved/loaded.
# For this notebook context, fitting on the original X ensures consistency.
loaded_scaler = StandardScaler()
loaded_scaler.fit(X) # X comes from cell a4a945dd and 91e3e399
print("StandardScaler re-instantiated and fitted successfully.")

# Re-instantiate and fit LabelEncoder to the original y data (assuming y is globally available)
# Similarly, the encoder object itself would typically be saved/loaded.
loaded_label_encoder = LabelEncoder()
loaded_label_encoder.fit(y) # y comes from cell a4a945dd and 91e3e399
print("LabelEncoder re-instantiated and fitted successfully.")

@app.route('/predict', methods=['POST'])
def predict():
    if request.is_json:
        try:
            data = request.get_json()
            # Convert the dictionary to a pandas DataFrame. It should be a single row.
            # Ensure the order of columns matches the training data features.
            # Assuming 'X' (from previous steps) is available to get column order.
            input_df = pd.DataFrame([data], columns=X.columns) # Use X.columns for consistency

            # Scale the input features using the loaded scaler
            input_scaled = loaded_scaler.transform(input_df)

            # Make predictions using the loaded model
            prediction_encoded = loaded_model.predict(input_scaled)

            # Decode the numerical prediction back to the original label
            prediction_label = loaded_label_encoder.inverse_transform(prediction_encoded)

            return jsonify({"prediction": prediction_label[0]})
        except Exception as e:
            return jsonify({"error": f"Error during prediction: {str(e)}"}), 400
    else:
        return jsonify({"error": "Request must be JSON"}), 400

print("Flask application with /predict endpoint and full prediction logic defined.")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

    # If running in Colab and needing public access
    from pyngrok import ngrok
    # Terminate open tunnels if any
    ngrok.kill()
    # Set up a tunnel to the Flask app
    public_url = ngrok.connect(addr='5000', proto='http')
    print(f"* ngrok tunnel available at: {public_url}")
