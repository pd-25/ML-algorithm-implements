from fastapi import FastAPI
import pickle
from pathlib import Path

from helper import transform_text
from schema import APIResponse, InputData

app = FastAPI(title="ML model implmentation through api")

MODEL_DIR = Path(__file__).resolve().parent.parent / "naive_bayes"

with open(MODEL_DIR / "vectorizer.pkl", "rb") as vectorizer_file:
    tfidf = pickle.load(vectorizer_file)
with open(MODEL_DIR / "model.pkl", "rb") as model_file:
    model = pickle.load(model_file)


@app.post("/predict", response_model=APIResponse)
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
