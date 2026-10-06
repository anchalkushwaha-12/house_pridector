import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# Path to trained model
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "house_price_model.joblib")

# Load model pipeline on startup
model_pipeline = None
try:
    if os.path.exists(MODEL_PATH):
        model_pipeline = joblib.load(MODEL_PATH)
        print(f"[Flask API] Model pipeline loaded successfully from {MODEL_PATH}")
    else:
        print(f"[Flask API] Warning: Model file not found at {MODEL_PATH}")
except Exception as e:
    print(f"[Flask API] Error loading model: {str(e)}")

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        "service": "House Price Prediction API",
        "version": "1.0",
        "status": "running",
        "endpoints": {
            "health": "GET /health",
            "predict": "POST /predict"
        }
    }), 200

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy" if model_pipeline is not None else "degraded",
        "model_loaded": model_pipeline is not None
    }), 200

@app.route('/predict', methods=['POST'])
def predict():
    if model_pipeline is None:
        return jsonify({
            "status": "error",
            "message": "Model pipeline is not loaded on server."
        }), 500

    data = request.get_json()
    if not data:
        return jsonify({
            "status": "error",
            "message": "No JSON payload provided in request."
        }), 400

    # Flexible key mapping for input keys
    key_mapping = {
        'area': 'Area(sqft)',
        'Area': 'Area(sqft)',
        'area_sqft': 'Area(sqft)',
        'Area(sqft)': 'Area(sqft)',
        'rooms': 'Rooms',
        'Rooms': 'Rooms',
        'bathrooms': 'Bathrooms',
        'Bathrooms': 'Bathrooms',
        'floors': 'Floors',
        'Floors': 'Floors',
        'location': 'Location',
        'Location': 'Location',
        'yearbuilt': 'YearBuilt',
        'year_built': 'YearBuilt',
        'YearBuilt': 'YearBuilt',
        'parking': 'Parking',
        'Parking': 'Parking',
        'condition': 'Condition',
        'Condition': 'Condition'
    }

    # Normalize payload
    normalized_data = {}
    for raw_key, value in data.items():
        if raw_key in key_mapping:
            normalized_data[key_mapping[raw_key]] = value
        else:
            normalized_data[raw_key] = value

    required_features = ['Area(sqft)', 'Rooms', 'Bathrooms', 'Floors', 'Location', 'YearBuilt', 'Parking', 'Condition']
    missing_features = [feat for feat in required_features if feat not in normalized_data]

    if missing_features:
        return jsonify({
            "status": "error",
            "message": f"Missing required features: {missing_features}"
        }), 400

    try:
        # Construct DataFrame for pipeline input
        input_df = pd.DataFrame([{
            'Area(sqft)': float(normalized_data['Area(sqft)']),
            'Rooms': int(normalized_data['Rooms']),
            'Bathrooms': int(normalized_data['Bathrooms']),
            'Floors': int(normalized_data['Floors']),
            'Location': str(normalized_data['Location']),
            'YearBuilt': int(normalized_data['YearBuilt']),
            'Parking': int(normalized_data['Parking']),
            'Condition': str(normalized_data['Condition'])
        }])

        # Make prediction
        predicted_price = float(model_pipeline.predict(input_df)[0])
        predicted_price = round(max(0, predicted_price), 2)
        
        area_sqft = float(normalized_data['Area(sqft)'])
        price_per_sqft = round(predicted_price / area_sqft, 2) if area_sqft > 0 else 0

        # Confidence interval approximation (~3.8% MAPE error margin)
        margin = predicted_price * 0.038

        return jsonify({
            "status": "success",
            "predicted_price": predicted_price,
            "formatted_price": f"${predicted_price:,.2f}",
            "price_per_sqft": price_per_sqft,
            "confidence_interval": {
                "lower_bound": round(max(0, predicted_price - margin), 2),
                "upper_bound": round(predicted_price + margin, 2)
            },
            "input_features": normalized_data
        }), 200

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Prediction failed: {str(e)}"
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting Flask server on http://127.0.0.1:{port}")
    app.run(host='127.0.0.1', port=port, debug=True)
