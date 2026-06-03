from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import json
from datetime import datetime

from app.model_loader import load_model

model = load_model()

app = FastAPI(
    title="Hospital Readmission Prediction API",
    description="Predict hospital readmission risk using Azure ML registered model",
    version="1.0"
)


class PatientData(BaseModel):
    age: float
    num_prior_admissions: float
    length_of_stay: float
    num_medications: float
    has_chronic_condition: float


@app.get("/")
def home():
    return {
        "message": "Hospital Readmission Prediction API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/predict")
def predict(data: PatientData):
    features = np.array([[
        data.age,
        data.num_prior_admissions,
        data.length_of_stay,
        data.num_medications,
        data.has_chronic_condition
    ]])

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    log_data = {
        "timestamp": str(datetime.now()),
        "inputs": data.model_dump(),
        "prediction": int(prediction),
        "readmission_probability": float(probability)
    }

    with open("prediction_logs.jsonl", "a") as f:
        f.write(json.dumps(log_data) + "\n")

    return {
        "prediction": int(prediction),
        "readmission_probability": round(float(probability), 4)
    }