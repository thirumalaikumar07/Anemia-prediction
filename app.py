from flask import Flask, request, jsonify
import pandas as pd
import joblib # Added joblib import
from sklearn.preprocessing import LabelEncoder, StandardScaler
app = Flask(__name__)
# Load the previously trained logistic regression model
loaded_model = joblib.load('logistic_regression_model.joblib')
print(f"Logistic Regression model loaded successfully from 'logistic_regression_model.joblib'")

# Re-instantiate and fit StandardScaler to the original X data
loaded_scaler = StandardScaler()
loaded_scaler.fit(X)
print("StandardScaler re-instantiated and fitted successfully.")

# Re-instantiate and fit LabelEncoder to the original y data
loaded_label_encoder = LabelEncoder()
loaded_label_encoder.fit(y)
print("LabelEncoder re-instantiated and fitted successfully.")

print("Type of loaded_model:", type(loaded_model))
print("Type of loaded_scaler:", type(loaded_scaler))
print("Type of loaded_label_encoder:", type(loaded_label_encoder))
# Create a Flask application instance (this line is intentionally not re-executing app = Flask(__name__)
# as it was defined in a previous cell)
# app = Flask(__name__)

# Define a route /predict that accepts POST requests
@app.route('/predict', methods=['POST'])
def predict():
    if request.is_json:
        try:
            data = request.get_json()
            # Convert the dictionary to a pandas DataFrame. It should be a single row.
            input_df = pd.DataFrame([data])

            # Ensure the input DataFrame has the same columns as the training data
            # This step is crucial if the input JSON might not contain all features or order them differently
            # For simplicity, assuming input_df directly matches X's columns
            # In a real-world scenario, you might want to reorder/fill missing columns

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

print("Flask application updated to include prediction logic.")


if __name__ =="__main__":
  print("Starting prediction API with preprocessing and model inference ...")
  app.run(debug=True)
