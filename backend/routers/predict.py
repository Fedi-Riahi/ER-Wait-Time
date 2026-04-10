

from fastapi  import APIRouter
from pydantic import BaseModel
from typing   import Any, Dict
import joblib, os, numpy as np

router     = APIRouter()
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'models')


class PredictRequest(BaseModel):
    model:  str = "random_forest"
    inputs: Dict[str, Any]


@router.post("/")
def predict(req: PredictRequest):
    path = os.path.join(MODELS_DIR, f"{req.model}_latest.joblib")

    if not os.path.exists(path):
        # Heuristic fallback when model not trained yet
        esi  = req.inputs.get("triage_level_ESI", 3)
        wait = {1:2, 2:10, 3:50, 4:95, 5:125}.get(int(esi), 50)
        wait += float(req.inputs.get("patients_waiting", 15)) * 0.8
        return {
            "wait_time_min": round(wait, 1),
            "model_used":    "heuristic",
            "warning":       f"Model '{req.model}' not trained yet via UI. Using heuristic.",
        }

    saved         = joblib.load(path)
    model         = saved["model"]
    scaler        = saved["scaler"]
    feature_names = saved["feature_names"]

    row = [float(req.inputs.get(f, 0)) for f in feature_names]
    X   = scaler.transform(np.array([row]))

    pred = model.predict(X)[0]
    return {
        "wait_time_min": round(float(pred), 1),
        "model_used":    req.model,
    }
