from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Any, Dict
import sys, os, joblib, numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import MODELS_DIR

router = APIRouter()

# Import registry to know which keys are valid
from model_registry_be import MODEL_REGISTRY


class PredictRequest(BaseModel):
    model:  str = "random_forest"
    inputs: Dict[str, Any]


@router.post("/")
def predict(req: PredictRequest):
    if req.model not in MODEL_REGISTRY:
        raise HTTPException(status_code=400, detail=f"Unknown model: '{req.model}'")

    path = os.path.join(MODELS_DIR, f"{req.model}_latest.joblib")

    if not os.path.exists(path):
        raise HTTPException(
            status_code=404,
            detail=(
                f"Model '{req.model}' has not been trained yet. "
                "Go to the Training page and train it first."
            ),
        )

    saved         = joblib.load(path)
    model         = saved["model"]
    scaler        = saved["scaler"]
    feature_names = saved["feature_names"]

    # Build input vector using the exact features the model was trained on.
    # Missing keys default to 0 — the frontend should send all required fields.
    missing = [f for f in feature_names if f not in req.inputs]
    row     = [float(req.inputs.get(f, 0)) for f in feature_names]
    X       = scaler.transform(np.array([row]))
    pred    = model.predict(X)[0]

    response = {
        "wait_time_min": round(float(pred), 1),
        "model_used":    req.model,
        "feature_names": feature_names,   # useful for the frontend to know what to send
    }
    if missing:
        response["warning"] = f"Missing inputs defaulted to 0: {missing}"

    return response


@router.get("/features")
def get_features(model: str = "random_forest"):
    """Return the feature names a trained model expects — useful for building the predict form."""
    path = os.path.join(MODELS_DIR, f"{model}_latest.joblib")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"'{model}' not trained yet.")
    saved = joblib.load(path)
    return {"model": model, "feature_names": saved["feature_names"]}
