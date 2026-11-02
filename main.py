
from pathlib import Path
from time import perf_counter
from typing import Literal

import json
import joblib
import pandas as pd

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "model" / "ckd_pipeline.joblib"
MODEL_INFO_PATH = BASE_DIR / "model" / "model_info.json"


# ============================================================
# 2. LOAD TRAINED MODEL
# ============================================================

model = None
model_info = {}
model_loaded = False

try:
    model = joblib.load(MODEL_PATH)

    with open(MODEL_INFO_PATH, "r") as f:
        model_info = json.load(f)

    model_loaded = True

    print("Model loaded successfully.")

except Exception as e:
    print(f"Model loading failed: {e}")


# ============================================================
# 3. CREATE FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="CKD Prediction API",
    description=(
        "Machine Learning REST API for "
        "Chronic Kidney Disease prediction"
    ),
    version="1.0.0"
)


# ============================================================
# 4. CORS CONFIGURATION
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500",
        "http://127.0.0.1:5500"
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# 5. API METRICS
# ============================================================

total_requests = 0
successful_predictions = 0
failed_requests = 0
total_response_time = 0.0


# ============================================================
# 6. REQUEST MONITORING MIDDLEWARE
# ============================================================

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):

    global total_requests
    global failed_requests
    global total_response_time

    total_requests += 1

    start_time = perf_counter()

    try:

        response = await call_next(request)

        if response.status_code >= 400:
            failed_requests += 1

        return response

    except Exception:

        failed_requests += 1
        raise

    finally:

        response_time = perf_counter() - start_time

        total_response_time += response_time


# ============================================================
# 7. INPUT DATA VALIDATION
# ============================================================

class PatientData(BaseModel):

    # Numerical features

    age: float = Field(..., ge=0, le=120)
    bp: float = Field(..., ge=0, le=250)
    sg: float = Field(..., ge=1.0, le=1.1)
    al: float = Field(..., ge=0, le=5)
    su: float = Field(..., ge=0, le=5)

    # Categorical features

    rbc: Literal["normal", "abnormal"]
    pc: Literal["normal", "abnormal"]
    pcc: Literal["present", "notpresent"]
    ba: Literal["present", "notpresent"]

    # Numerical features

    bgr: float = Field(..., ge=0)
    bu: float = Field(..., ge=0)
    sc: float = Field(..., ge=0)
    sod: float = Field(..., ge=0)
    pot: float = Field(..., ge=0)
    hemo: float = Field(..., ge=0)
    pcv: float = Field(..., ge=0, le=100)
    wc: float = Field(..., ge=0)
    rc: float = Field(..., ge=0)

    # Categorical features

    htn: Literal["yes", "no"]
    dm: Literal["yes", "no"]
    cad: Literal["yes", "no"]
    appet: Literal["good", "poor"]
    pe: Literal["yes", "no"]
    ane: Literal["yes", "no"]


# ============================================================
# 8. GET /
# ============================================================

@app.get("/")
def root():

    return {
        "message": "CKD Prediction API",
        "version": "1.0.0",
        "description": (
            "REST API for Chronic Kidney Disease prediction"
        )
    }


# ============================================================
# 9. GET /health
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


# ============================================================
# 10. GET /ready
# ============================================================

@app.get("/ready")
def ready():

    if model_loaded:

        return {
            "ready": True,
            "model_loaded": True
        }

    return JSONResponse(
        status_code=503,
        content={
            "ready": False,
            "model_loaded": False
        }
    )


# ============================================================
# 11. GET /model-info
# ============================================================

@app.get("/model-info")
def model_information():

    return model_info


# ============================================================
# 12. POST /predict
# ============================================================

@app.post("/predict")
def predict(patient: PatientData):

    global successful_predictions

    if not model_loaded:

        return JSONResponse(
            status_code=503,
            content={
                "error": "Model is not loaded"
            }
        )

    try:

        # Convert Pydantic object into dictionary

        patient_data = patient.model_dump()

        # Convert dictionary into DataFrame

        input_df = pd.DataFrame([patient_data])

        # Generate prediction

        prediction = int(
            model.predict(input_df)[0]
        )

        # Generate probabilities

        probabilities = model.predict_proba(input_df)[0]

        # Probability for each class

        not_ckd_probability = float(probabilities[0])
        ckd_probability = float(probabilities[1])

        # Convert numerical prediction into result

        if prediction == 1:

            result = "CKD"
            predicted_probability = ckd_probability

        else:

            result = "Not CKD"
            predicted_probability = not_ckd_probability

        # Update successful prediction count

        successful_predictions += 1

        # Return API response

        return {
            "prediction": result,

            "model_output_probability": round(
                predicted_probability, 4
            ),

            "ckd_probability": round(
                ckd_probability, 4
            ),

            "not_ckd_probability": round(
                not_ckd_probability, 4
            ),

            "model_version": model_info.get(
                "version", "unknown"
            ),

            "note": (
                "Model output only; "
                "not a medical diagnosis."
            )
        }

    except Exception as e:

        return JSONResponse(
            status_code=500,
            content={
                "error": "Prediction failed",
                "details": str(e)
            }
        )


# ============================================================
# 13. GET /metrics
# ============================================================

@app.get("/metrics")
def metrics():

    average_response_time_ms = 0

    if total_requests > 0:

        average_response_time_ms = (
            total_response_time / total_requests
        ) * 1000

    return {
        "total_requests": total_requests,

        "successful_predictions": successful_predictions,

        "failed_requests": failed_requests,

        "average_response_time_ms": round(
            average_response_time_ms, 2
        )
    }
