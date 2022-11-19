from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle
from flask import url_for
app = Flask(__name__)
model_ai=pickle.load(open('Hridayai1.pkl','rb'))

@app.route('/')
def home():
    return render_template('main.html')

@app.route('/predict', methods=["GET","POST"])
def predict():
    if request.method == "POST":
        age = int(request.form["age"])

        sex = int(request.form["Sex"])

        cp = int(request.form["cp"])

        trestbps = int(request.form["trestbps"])

        chol = int(request.form["chol"])

        fbs = int(request.form["fbs"])

        restecg = int(request.form["restecg"])

        thalach = int(request.form["thalach"])

        exang = int(request.form["exang"])

        oldpeak = float(request.form["oldpeak"])

        slope = int(request.form["slope"])

        ca = int(request.form["ca"])

        thal = int(request.form["thal"])

        data = [[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]]
        out1 = pd.DataFrame(data, columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang','oldpeak','slope','ca','thal'])
        heart_prediction = model_ai.predict(data)
        #print(sample)
        # heart_pred = h.heart_disease(sample)
        #print(heart_pred)
        #hp = heart_pred
        return render_template('result.html',prediction = heart_prediction)
       

# @app.route("/sub", methods = ["POST"])
# def submit():
#     if request.method == "POST":
#         age = request.form["age"]
#
#     return render_template("sub.html", age=age)


if __name__ == '__main__':
    app.run(debug=True)