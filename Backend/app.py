import io
from flask import Flask, request, jsonify
import joblib
import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import nltk
import os
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Path to model file
model_path = "logistic_regression_model.pkl"

# Check if the model file exists and load the model
try:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found. Please check the path: {model_path}")
    
    model = joblib.load(model_path)
    print("Model loaded successfully.")
except Exception as e:
    raise RuntimeError(f"Failed to load model: {str(e)}")

# Download NLTK stopwords if not already downloaded
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))
porter_stemmer = PorterStemmer()

# Preprocessing function
def preprocess_text(text):
    """
    Preprocess the input text by removing non-alphabet characters,
    converting to lowercase, removing stopwords, and applying stemming.
    """
    # Remove non-alphabetic characters
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    # Convert to lowercase and split into words
    words = text.lower().split()
    # Remove stopwords and apply stemming
    stemmed_words = [porter_stemmer.stem(word) for word in words if word not in stop_words]
    # Join the processed words back into a single string
    return ' '.join(stemmed_words)

# Define the predict route
@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle POST requests for text prediction.
    The input is expected to be JSON with a 'text' field.
    """
    # Parse JSON input
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'Invalid input, expected JSON with a "text" field'}), 400

    input_text = data.get('text', '').strip()
    if not input_text:
        return jsonify({'error': 'Empty text provided'}), 400

    # Preprocess the input text
    preprocessed_text = preprocess_text(input_text)

    try:
        # Predict using the loaded model
        prediction = model.predict([preprocessed_text])[0]

        # Map prediction result to human-readable format
        result = 'real' if prediction == 1 else 'fake'
        return jsonify({'prediction': result})
    except Exception as e:
        # Handle prediction errors gracefully
        return jsonify({'error': f"Prediction failed: {str(e)}"}), 500

# Entry point for the app (suitable for Render or local deployment)
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Default Render port is 5000
    app.run(host='0.0.0.0', port=port, debug=True)
