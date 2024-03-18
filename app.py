from flask import Flask, render_template, request
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

if __name__ == "__main__":
    app.run(debug=True)