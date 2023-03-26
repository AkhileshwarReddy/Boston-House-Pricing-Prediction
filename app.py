import pickle
import numpy as np
import pandas as pd

from flask import Flask, request, jsonify, url_for, render_template

## Create the flask app
app = Flask(__name__)

## Load the model and scaler
regmodel = pickle.load(open('regmodel.pkl', 'rb'))
scaler = pickle.load(open('scaling.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    return jsonify(__predict(data))

@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    return render_template("home.html", prediction_text=f"The house price prediction is {__predict(data)}")

def __predict(data):
    data_point = np.array(list(data)).reshape(1, -1)
    new_data = scaler.transform(data_point)
    output = regmodel.predict(new_data)
    return output[0]


if __name__ == '__main__':
    app.run(debug=True)
