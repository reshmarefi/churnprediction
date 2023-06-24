from flask import Flask,request,render_template,url_for
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib as joblib


model=joblib.load('model.pkl')

#Let's create the flask app reshma
app = Flask(__name__)

#To load the model.pkl
#model = pickle.load(open("model.pkl","rb"))

@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/predict",methods=["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    #sl=request.form['SepalLength']
    #sw = request.form['SepalWidth']
    #pl = request.form['PetalLength']
    #pw = request.form['PetalWidth']
    #data = np.array([[sl, sw, pl, pw]])
    #x = scaler.transform(data)
    #print(x)
    #prediction = model.predict(data)
    return render_template("index.html", prediction_text = (prediction[0]))

if __name__ == '__main__':
    app.run(debug = True)
