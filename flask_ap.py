from flask import Flask, request, jsonify
import pandas as pd
import pickle
import os
import requests
from flask_cors import CORS

# URL of your model stored on Google Drive
MODEL_URL = 'https://drive.google.com/uc?export=download&id=1D1kPHNLC1MpVirOp-jhU3ViXkDJVUS_N'
MODEL_PATH = 'fantasy_score_model.pkl'

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model from Google Drive...")
        with requests.Session() as session:
            response = session.get(MODEL_URL, stream=True)
            # Check if Google Drive sends a 'confirm' link for large files
            if "text/html" in response.headers.get("content-type", ""):
                # Parse the confirmation page and extract the download link
                for key, value in response.cookies.items():
                    if key.startswith('download_warning'):
                        MODEL_URL_confirm = MODEL_URL + "&confirm=" + value
                        response = session.get(MODEL_URL_confirm, stream=True)
                        break
            # Write the model file locally
            with open(MODEL_PATH, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:  # Filter out keep-alive new chunks
                        f.write(chunk)
            print("Model downloaded successfully.")

# Ensure the model is downloaded before starting the app
download_model()

# Load your data and model
df = pd.read_csv('fantasy_scores.csv')
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

# Create Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Enable CORS for the entire Flask app

@app.route('/')
def home():
    return "Hello, World!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_data = pd.DataFrame({
        'venue': [data['venue']],
        'player_name': [data['player_name']],
        'player_team': [data['player_team']],
        'batting_first_team': [data['batting_first_team']],
        'bowling_first_team': [data['bowling_first_team']]
    })

    input_data_encoded = pd.get_dummies(input_data)
    input_data_encoded = input_data_encoded.reindex(columns=model.feature_names_in_, fill_value=0)

    predicted_score = model.predict(input_data_encoded)

    return jsonify({'predicted_score': predicted_score[0]})

@app.route('/options', methods=['GET'])
def options():
    return jsonify({
        'venues': df['venue'].unique().tolist(),
        'player_names': df['player_name'].unique().tolist(),
        'player_teams': df['player_team'].unique().tolist(),
        'batting_first_teams': df['batting_first_team'].unique().tolist(),
        'bowling_first_teams': df['bowling_first_team'].unique().tolist()
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True, use_reloader=False)
