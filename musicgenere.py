import joblib
import librosa
import numpy as np
import pandas as pd
from feature import Feature
import flask
from flask import request, jsonify, render_template
import os
# Load the trained model and scaler
model = joblib.load('/home/kushagra/Documents/code/AI/project/music_genere_classification/best_rf_model.pkl')
scaler = joblib.load('/home/kushagra/Documents/code/AI/project/music_genere_classification/scaler.pkl')

# Define the genre dictionary
dict1 = {0: 'blues', 1: 'classical', 2: 'country', 3: 'disco', 4: 'hiphop', 5: 'jazz', 6: 'metal', 7: 'pop', 8: 'reggae', 9: 'rock'}

# Function to extract features from audio file
def extract_features(file_name):
    try:
        feature_instance = Feature(file_name)
        features = feature_instance.extract_features()
        features = [features]
        scaler_f = scaler.transform(features)
        answers = model.predict(scaler_f)
        answer = dict1[answers[0]]
    except Exception as e:
        print(f"Error encountered while parsing file: {file_name}")
        print(e)
        return None
    return answer

app = flask.Flask(__name__)

@app.route('/')
def home():
    print(os.listdir())
    return render_template('main.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file:
        file.save(file.filename)
        answer = extract_features(file.filename)
        if answer:
            return jsonify({'genre': answer})
        else:
            return jsonify({'error': 'Error processing the file'})
    else:
        return jsonify({'error': 'File not allowed'})

if __name__ == "__main__":
    app.run(debug=True)
