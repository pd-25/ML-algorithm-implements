# ML Algorithm Implements

This repository is a collection of machine learning model implementations and FastAPI endpoints for practicing and demonstrating ML algorithms in a real-world API setup.

It includes examples from:
- Supervised learning
- Unsupervised learning
- Self-supervised learning
- Semi-supervised learning
- Reinforcement learning

The current implementation focuses on two practical supervised learning examples:
- Naive Bayes for SMS spam detection
- Logistic Regression for credit card fraud detection

## Live API Documentation

Open the deployed Swagger UI here:

https://ml-algorithm-implements.onrender.com/docs#/

## Project Structure

```text
ML-algorithm-implements/
├── application/
│   ├── __init__.py
│   ├── helper.py
│   ├── main.py
│   └── schema.py
├── logistic_regression/
│   ├── credit_card_fraud_detection.ipynb
│   └── creditcard.csv
├── naive_bayes/
│   ├── naive_bayes_model_create.ipynb
│   ├── spam_sms_naive_byes.ipynb
│   ├── spam.csv
│   ├── model.pkl
│   └── vectorizer.pkl
├── pyproject.toml
├── README.md
└── .venv/
```

## Tech Stack

- Python
- FastAPI
- scikit-learn
- pandas
- numpy
- NLTK
- Pickle model serialization

## API Endpoints

### Health Check

- GET /health

Returns the server health status.

### Naive Bayes Spam Prediction

- POST /naive-predict

Request body:

```json
{
  "input_str": "Congratulations! You have won a free prize. Claim now."
}
```

Response:

```json
{
  "success": true,
  "message": "This is spam"
}
```

### Logistic Regression Fraud Prediction

- POST /logistic-reg-predict

Request body example:

```json
{
  "Time": 0.0,
  "V1": -1.359807,
  "V2": -0.072781,
  "V3": 2.536347,
  "V4": 1.378155,
  "V5": -0.338321,
  "V6": 0.462388,
  "V7": 0.239599,
  "V8": 0.098698,
  "V9": 0.363787,
  "V10": 0.090794,
  "V11": -0.5516,
  "V12": -0.617801,
  "V13": -0.99139,
  "V14": -0.311169,
  "V15": 1.468177,
  "V16": -0.470401,
  "V17": 0.207971,
  "V18": 0.025791,
  "V19": 0.403993,
  "V20": 0.251412,
  "V21": -0.018307,
  "V22": 0.277838,
  "V23": -0.110474,
  "V24": 0.066928,
  "V25": 0.128539,
  "V26": -0.189115,
  "V27": 0.133558,
  "V28": -0.021053,
  "Amount": 149.62
}
```

Response:

```json
{
  "success": true,
  "message": "This is a fraud transcation"
}
```

## Local Setup

1. Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -e .
```

3. Run the app:

```bash
uvicorn application.main:app --reload
```

4. Open the API docs locally:

```text
http://127.0.0.1:8000/docs
```

## Notes

This project is mainly for learning and demonstration. It combines model training notebooks with a deployment-ready FastAPI service so that trained ML models can be tested through HTTP requests.

## License

This project is intended for educational and experimentation purposes.
