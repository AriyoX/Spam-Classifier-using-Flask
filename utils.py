import pickle

cv = pickle.load(open("models/cv.pkl", "rb")) # tokenizer
clf = pickle.load(open("models/clf.pkl", "rb")) # model

def make_prediction(email):
    tokenized_email = cv.transform([email])
    prediction = clf.predict(tokenized_email)
    prediction = 1 if prediction == 1 else -1
    return prediction
