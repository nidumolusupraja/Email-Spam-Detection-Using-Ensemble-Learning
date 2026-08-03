# app.py
from flask import Flask, request, render_template, jsonify
import xgboost as xgb
from joblib import load
import numpy as np

app = Flask(__name__)
model = load('model.pkl')
tfidf = load('tfidf.pkl')

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    message = request.form["email"]
    vect_msg = tfidf.transform([message])
    prediction = model.predict(vect_msg)[0]
    result = "Spam 🚨" if prediction == 0 else "Ham ✅"
    return render_template("index.html", result=result, email=message)

if __name__ == "__main__":
    app.run(debug=True)
