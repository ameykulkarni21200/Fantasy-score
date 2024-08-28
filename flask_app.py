from flask import Flask, request, jsonify
import pandas as pd
import pickle

# Load your data and model
df = pd.read_csv('fantasy_scores.csv')
with open('fantasy_score_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Create Flask app
app = Flask(__name__)

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

    # One-hot encode the input data to match the model's expected format
    input_data_encoded = pd.get_dummies(input_data)
    input_data_encoded = input_data_encoded.reindex(columns=model.feature_names_in_, fill_value=0)

    # Make prediction
    predicted_score = model.predict(input_data_encoded)

    # Return the predicted score as a JSON response
    return jsonify({'predicted_score': predicted_score[0]})

@app.route('/options', methods=['GET'])
def get_options():
    options = {
        'venues': df['venue'].unique().tolist(),
        'player_names': df['player_name'].unique().tolist(),
        'player_teams': df['player_team'].unique().tolist(),
        'batting_first_teams': df['batting_first_team'].unique().tolist(),
        'bowling_first_teams': df['bowling_first_team'].unique().tolist()
    }
    return jsonify(options)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
