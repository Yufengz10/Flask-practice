from flask import Flask, jsonify, request, render_template
import json
import numpy as np
import pickle


with open("model.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)


@app.route('/', methods=["GET", "POST"])
def index():
    pred = ""
    if request.method == "POST":
        review_aroma = request.form["review_aroma"]
        review_appearance = request.form["review_appearance"]
        review_palate = request.form["review_palate"]
        review_taste = request.form["review_taste"]
        beer_abv = request.form["beer_abv"]
        style_enc = request.form["style_enc"]
        
        X = np.array([[float(review_aroma), float(review_appearance), float(review_palate), float(review_taste), float(beer_abv), int(style_enc)]])
        pred = model.predict(X)
    return render_template("index.html", pred=pred)


if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=5000)
