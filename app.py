import numpy as np

import numpy as np
import pickle
from flask import Flask, render_template, request

app = Flask(__name__)

# Load trained model
try:
    model = pickle.load(open("model.pkl", "rb"))
except:
    model = None


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        jf = float(request.form["JF"])
        mam = float(request.form["MAM"])
        jjas = float(request.form["JJAS"])
        ond = float(request.form["OND"])

        # IMPORTANT: Order must match training
        features = np.array([[jf, mam, jjas, ond]])

        if model:
            prediction = model.predict(features)[0]
        else:
            return "Model not loaded."

        if prediction == 1:
            return render_template("chance.html")
        else:
            return render_template("nochance.html")

    except Exception as e:
        return f"Error: {e}"


if __name__ == "__main__":
    app.run(debug=True)
from flask import Flask, render_template
app = Flask(__name__)

@app.route("/")

def index():
    return render_template("index.html")


@app.route("/chance")

def chance():
    return render_template("chance.html")
    

@app.route("/nochance")
def nochance():

    
    return render_template("nochance.html")


if __name__ == "__main__":
    app.run(debug=True)
