# app.py

from flask import Flask, render_template, request
from joblib import load
import numpy as np

app = Flask(__name__)
model = load('random_forest_model.joblib')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    sales = float(request.form['sales'])
    segment = int(request.form['segment'])
    category = int(request.form['category'])

    prediction = model.predict([[sales, segment, category]])
    return render_template('index.html', prediction=f'${round(prediction[0], 2)}')

if __name__ == '__main__':
    app.run(debug=True)
