import numpy as np
import pandas as pd
from flask import Flask , render_template , request
# from sklearn.linear_model import LinearRegression
import pickle
app = Flask(__name__)

@app.route("/")
def fun1():
    return render_template("index.html")



@app.route("/predict",methods = ['GET','POST'])
def fun3():
    a = [float(i) for i in request.form.values()]
    b = [np.array(a)]
    sol = pickle.load(open("calories.pkl",'rb'))
    predictions = sol.predict(b)
    predictions = predictions[0]
    return render_template('index.html',prediction_text = predictions)




if __name__ == '__main__':
    app.run(debug=True)