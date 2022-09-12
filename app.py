from os import path
import pandas as pd
import numpy as np
import flask
import joblib
from flask import Flask, render_template, request

app=Flask(__name__)

@app.route('/')

def index():
    return flask.render_template('index.html')

def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1,6)
    joblib_path = path.join(path.dirname(path.abspath(__file__)), "heart.joblib")
    print(joblib_path)
    loaded_model = joblib.load(joblib_path)
    result = loaded_model.predict(to_predict)
    return 'You are safe' if int(result[0]) else 'You have a risk of heart disease'

@app.route('/predict',methods = ['POST'])

def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        if to_predict_list[-1] == '1':
            to_predict_list += ["0"]
        else :
            to_predict_list += ["1"]
        try:
            to_predict_list = list(map(float, to_predict_list))
        except:
            return render_template("index.html",prediction="Enter complete data")
    result = ValuePredictor(to_predict_list)
    prediction = str(result)
    return render_template("index.html",prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)