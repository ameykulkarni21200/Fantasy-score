from flask import Flask, request, jsonify
import pandas as pd
import pickle
from flask_cors import CORS  # Import the CORS module

# Load your data and model
#df = pd.read_csv('F:\\fantasy_scores.csv')
df = pd.read_csv('fantasy_scores.csv')
with open('fantasy_score_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Create Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Enable CORS for the entire Flask app


@app.route('/')
def home():
    return "Hello, World!"







@app.route('/', methods=['POST'])
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

@app.route('/', methods=['GET'])
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

