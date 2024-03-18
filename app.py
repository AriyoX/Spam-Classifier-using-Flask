from flask import Flask, render_template, request, jsonify
import pickle

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

    tokenized_email = cv.transform([email])
    prediction = clf.predict(tokenized_email)
    prediction = 1 if prediction == 1 else -1
    return render_template("index.html", prediction=prediction, email=email)

@app.route("/api/predict", methods = ["POST"])
def api_predict():
    data = request.get_json(force = True)
    email = data["content"]
    tokenized_email = cv.transform([email])
    prediction = clf.predict(tokenized_email)
    prediction = 1 if prediction == 1 else -1
    return jsonify({prediction: prediction})

if __name__ == "__main__":
    app.run(debug=True)