from fastapi import FastAPI
import pickle
from pathlib import Path

from helper import transform_text
from schema import APIResponse, InputData, LogisticRegRequest

app = FastAPI(title="ML model implmentation through api")

MODEL_DIR = Path(__file__).resolve().parent.parent

with open(MODEL_DIR / "naive_bayes/vectorizer.pkl", "rb") as vectorizer_file:
    tfidf = pickle.load(vectorizer_file)
with open(MODEL_DIR / "naive_bayes/model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open(MODEL_DIR / "logistic_regression/logistic_model.pkl", "rb") as logistic_file:
    logistic_model = pickle.load(logistic_file)

@app.post("/naive-predict", response_model=APIResponse)
def predict_input(input_text: InputData):
    # Preprocess text
    transformed_text = transform_text(input_text.input_str)

    # Vectorize
    vectorized_text = tfidf.transform([transformed_text])
    # print(vectorized_text)
    # model predict
    result = model.predict(vectorized_text)[0]
    # response
    msg = ""
    if result == 1:
        msg = "This is spam"
    else:
        msg = "Not Spam"

    return APIResponse(success=True, message=msg)

@app.post("/logistic-reg-predict", response_model=APIResponse)
def predict_logistic_reg(request_input: LogisticRegRequest):
    print(request_input)
    return "l"
    result = logistic_model.predict(request_input)[0]
    print(result)
    return result
    