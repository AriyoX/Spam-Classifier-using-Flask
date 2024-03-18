from flask import Flask, render_template, request, jsonify
import pickle
from utils import make_prediction

cv = pickle.load(open("models/cv.pkl", "rb")) # tokenizer
clf = pickle.load(open("models/clf.pkl", "rb")) # model

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        email  = request.form.get("content")
    prediction = make_prediction(email)
    return render_template("index.html", prediction=prediction, email=email)

@app.route("/api/predict", methods = ["POST"])
def api_predict():
    data = request.get_json(force = True)
    email = data["content"]
    prediction = make_prediction(email)
    return jsonify({prediction: prediction})

if __name__ == "__main__":
    app.run(debug=True)