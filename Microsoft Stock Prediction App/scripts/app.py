from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# Load models
LR = joblib.load('models/linear_regression.pkl')
RF = joblib.load('models/random_forest.pkl')
xg = joblib.load('models/xgboost.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    date = data.get('date')
    feature_values = data.get('features')

    # Example prediction (replace with actual feature extraction logic)
    prediction = LR.predict([feature_values])[0]

    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
