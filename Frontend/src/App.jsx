import { useState } from 'react';
import axios from 'axios';

function App() {
  const [text, setText] = useState('');
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);

  const handlePredict = async () => {
    if (!text) {
      alert('Please enter some text to analyze.');
      return;
    }

    setLoading(true);
    setPrediction(null);

    try {
      const response = await axios.post('http://127.0.0.1:5000/predict', { text });
      setPrediction(response.data.prediction);
    } catch (error) {
      console.error("Error connecting to the prediction API:", error);
      alert("Failed to get a response from the server.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-500">
      <div className="bg-gray-400 p-8 rounded shadow-md w-full max-w-md">
        <h1 className="text-2xl font-bold mb-4 text-center">Fake News Detector</h1>
        <textarea
          className="w-full p-4 border rounded-md resize-none focus:outline-none focus:ring-2 focus:ring-blue-400"
          rows="5"
          placeholder="Enter the news text here..."
          value={text}
          onChange={(e) => setText(e.target.value)}
        ></textarea>
        <button
          className="w-full mt-4 py-2 px-4 bg-blue-500 text-white font-semibold rounded hover:bg-blue-600"
          onClick={handlePredict}
          disabled={loading}
        >
          {loading ? 'Predicting...' : 'Predict'}
        </button>
        {prediction && (
          <div className="mt-4 p-4 bg-gray-200 rounded text-center">
            <span className="font-bold">Prediction: </span>
            <span className={prediction === 'real' ? 'text-green-500' : 'text-red-500'}>
              {prediction === 'real' ? 'Real' : 'Fake'}
            </span>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
