from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

# Load the saved model and scaler
model = joblib.load('Cell_Anomaly_Classifier.pkl') 
scaler = joblib.load('scaler.pkl')   


app = FastAPI()

class ClassifierFeatures(BaseModel):
    charge_CCE: float
    charge_PRB: float
    rsrp: float
    sinr: float
    ho_success_rate: float
    call_drop_rate_pct: float

@app.get("/")
def home():
    return {
        "message": "Welcome to the cell anomaly classifier API! Use the /predict endpoint to predict cells anomaly types"
    }


# Define the prediction endpoint
@app.post("/predict")
def predict(Classifier: ClassifierFeatures):
    # Extract the features from the incoming request
    features = np.array([
        [
            Classifier.charge_CCE,
            Classifier.charge_PRB,
            Classifier.rsrp,
            Classifier.sinr,
            Classifier.ho_success_rate,
            Classifier.call_drop_rate_pct,
            
        ]
    ])
    # Scale the input features using the saved scaler
    scaled_features = scaler.transform(features)


    # Make the prediction using the loaded model
    prediction = model.predict(scaled_features)
    
    # Return the prediction (wine quality)
    return {"predicted_status": str(prediction[0])}