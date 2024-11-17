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

# Absolute path to model file
model_path = "fake_news_model.pkl"

# Check if the model file exists
if os.path.exists(model_path):
    try:
        model = joblib.load(model_path)
        print("Model loaded successfully.")
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}")
else:
    raise FileNotFoundError(f"Model file not found. Please check the path: {model_path}")

# Download stopwords if not already downloaded
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))
porter_stemmer = PorterStemmer()

# Define the function to preprocess the input text
def preprocess_text(text):
    # Remove non-alphabet characters
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    # Convert to lowercase and split
    words = text.lower().split()
    # Apply stemming and remove stopwords
    stemmed_words = [porter_stemmer.stem(word) for word in words if word not in stop_words]
    # Join words back to form a single string
    return ' '.join(stemmed_words)

# Define the predict route
@app.route('/predict', methods=['POST'])
def predict():
    # Get data from request
    data = request.get_json()
    
    # Validate the input
    if not data or 'text' not in data:
        return jsonify({'error': 'Invalid input, expected JSON with a "text" field'}), 400
    
    input_text = data.get('text', '')

    # Preprocess the input text
    preprocessed_text = preprocess_text(input_text)
    
    try:
        # Make prediction
        prediction = model.predict([preprocessed_text])

        # Prepare and send response
        result = 'real' if prediction[0] == 1 else 'fake'
        return jsonify({'prediction': result})
    except Exception as e:
        return jsonify({'error': f"Prediction failed: {str(e)}"}), 500

# Entry point for Render deployment
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Default Render port is 5000
    app.run(host='0.0.0.0', port=port, debug=True)
