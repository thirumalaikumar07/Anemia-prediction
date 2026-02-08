import joblib

# Define the filename for the exported model
model_filename = 'logistic_regression_model.joblib'

# Export the trained logistic regression model
joblib.dump(logistic_model, model_filename)

print(f"Logistic Regression model exported successfully as '{model_filename}'")
import joblib

# Define the filename for the exported scaler
scaler_filename = 'scaler.joblib'

# Export the trained scaler
joblib.dump(loaded_scaler, scaler_filename)

print(f"StandardScaler exported successfully as '{scaler_filename}'")
import joblib

# Define the filename for the exported LabelEncoder
label_encoder_filename = 'label_encoder.joblib'

# Export the fitted LabelEncoder
joblib.dump(loaded_label_encoder, label_encoder_filename)

print(f"LabelEncoder exported successfully as '{label_encoder_filename}'")
import joblib

# Define the filename for the exported feature column names
feature_columns_filename = 'feature_columns.joblib'

# Export the feature column names
joblib.dump(X.columns.tolist(), feature_columns_filename)

print(f"Feature column names exported successfully as '{feature_columns_filename}'")
get_ipython().run_cell_magic('writefile', 'app.py', """import pandas as pd\nfrom flask import Flask, request, jsonify\nimport joblib\nimport os\n\n# Define filenames for the saved components\nMODEL_FILENAME = 'logistic_regression_model.joblib'\nSCALER_FILENAME = 'scaler.joblib'\nLABEL_ENCODER_FILENAME = 'label_encoder.joblib'\nFEATURE_COLUMNS_FILENAME = 'feature_columns.joblib'\n\n# Load the trained model, scaler, label encoder, and feature column names\ntry:\n    loaded_model = joblib.load(MODEL_FILENAME)\n    loaded_scaler = joblib.load(SCALER_FILENAME)\n    loaded_label_encoder = joblib.load(LABEL_ENCODER_FILENAME)\n    feature_columns = joblib.load(FEATURE_COLUMNS_FILENAME)\n    print("All necessary model components loaded successfully.")\nexcept Exception as e:\n    print(f"Error loading model components: {e}")\n    # In a real application, you might want to handle this error more robustly\n    # For now, we'll assume successful loading or fail early during deployment.\n    loaded_model = None\n    loaded_scaler = None\n    loaded_label_encoder = None\n    feature_columns = []\n\napp = Flask(__name__)\n\n@app.route('/predict', methods=['POST'])\ndef predict():\n    if loaded_model is None or loaded_scaler is None or loaded_label_encoder is None or not feature_columns:\n        return jsonify({"error": "Model components not loaded. Please check server logs."}), 500\n\n    if request.is_json:\n        try:\n            data = request.get_json()\n\n            # Convert input data to a pandas DataFrame, ensuring column order\n            input_df = pd.DataFrame([data], columns=feature_columns)\n\n            # Scale the input features\n            input_scaled = loaded_scaler.transform(input_df)\n\n            # Make prediction\n            prediction_encoded = loaded_model.predict(input_scaled)\n\n            # Decode the prediction back to original label\n            prediction_label = loaded_label_encoder.inverse_transform(prediction_encoded)\n\n            return jsonify({"prediction": prediction_label[0]})\n        except Exception as e:\n            return jsonify({"error": f"Error during prediction: {str(e)}"}), 400\n    else:\n        return jsonify({"error": "Request must be JSON"}), 400\n\nif __name__ == '__main__':\n    # For local execution, run: python app.py\n    # For Colab, further steps are needed to expose it publicly (e.g., ngrok)\n    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)\n""")
print("app.py created successfully with full prediction logic.")
get_ipython().system('pip install pyngrok')
print("pyngrok installed successfully.")
from pyngrok import ngrok

# IMPORTANT: Replace 'YOUR_AUTHTOKEN_HERE' with your actual ngrok authtoken
# You can find your authtoken at https://dashboard.ngrok.com/get-started/your-authtoken
ngrok.set_auth_token('YOUR_AUTHTOKEN_HERE') # Make sure to replace this!
get_ipython().system('python app.py &')
print("Flask application 'app.py' started in the background.")
from pyngrok import ngrok

# Terminate open tunnels if any (important for Colab to avoid conflicts)
ngrok.kill()

# Set up a tunnel to the Flask app running on port 5000
public_url = ngrok.connect(addr='5000', proto='http')
print(f"* ngrok tunnel available at: {public_url}")
print("You can access your Flask API publicly via the URL above.")


print("ngrok authtoken set. If this is your first time, you may need to register your authtoken once.")
