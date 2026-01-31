from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import logging
from dotenv import load_dotenv
import os

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------
# Load environment variables
# ----------------------------
load_dotenv()

# Optionnel: utiliser des variables d'environnement
MODEL_PATH = os.getenv("MODEL_PATH", "cell_anomaly_classifier.pkl")
SCALER_PATH = os.getenv("SCALER_PATH", "scaler.pkl")

# ----------------------------
# Globals (preloaded at startup)
# ----------------------------
model = None
scaler = None

# ----------------------------
# FastAPI app
# ----------------------------
app = FastAPI()

# ----------------------------
# Startup event: preload model/scaler
# ----------------------------
@app.on_event("startup")
def load_assets():
    global model, scaler
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        logger.info("Model and scaler loaded successfully.")
        logger.info(f"MODEL_PATH={MODEL_PATH} | SCALER_PATH={SCALER_PATH}")
    except Exception as e:
        logger.exception(f"Error loading model or scaler: {e}")
        # On bloque dès le démarrage si c'est critique
        raise RuntimeError("Failed to load model/scaler. Check files/paths.") from e

# ----------------------------
# Request schema
# ----------------------------
class ClassifierFeatures(BaseModel):
    charge_CCE: float
    charge_PRB: float
    rsrp: float
    sinr: float
    ho_success_rate: float
    call_drop_rate_pct: float

# ----------------------------
# Routes
# ----------------------------
@app.get("/")
def home():
    return {
        "message": "Welcome to the cell anomaly classifier API! Use POST /predict to predict anomaly type."
    }

@app.post("/predict")
def predict(payload: ClassifierFeatures):
    if model is None or scaler is None:
        logger.error("Model or scaler not loaded.")
        raise HTTPException(status_code=503, detail="Model not loaded. Please try again later.")

    # Validation simple: certains KPIs doivent être >= 0
    # (rsrp/sinr peuvent être négatifs, donc on ne les bloque pas)
    if (
        payload.charge_CCE < 0
        or payload.charge_PRB < 0
        or payload.ho_success_rate < 0
        or payload.call_drop_rate_pct < 0
    ):
        logger.warning("Invalid input: some values are negative.")
        raise HTTPException(status_code=400, detail="charge_CCE/charge_PRB/ho_success_rate/call_drop_rate_pct must be >= 0")

    # Prépare l'input dans le même ordre que le training
    features = np.array([[
        payload.charge_CCE,
        payload.charge_PRB,
        payload.rsrp,
        payload.sinr,
        payload.ho_success_rate,
        payload.call_drop_rate_pct
    ]], dtype=float)

    try:
        scaled_features = scaler.transform(features)
        prediction = model.predict(scaled_features)

        logger.info(
            "Prediction | CCE=%s PRB=%s RSRP=%s SINR=%s HO=%s DROP=%s => %s",
            payload.charge_CCE, payload.charge_PRB, payload.rsrp, payload.sinr,
            payload.ho_success_rate, payload.call_drop_rate_pct, prediction[0]
        )

        return {"predicted_status": str(prediction[0])}

    except Exception as e:
        logger.exception(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))