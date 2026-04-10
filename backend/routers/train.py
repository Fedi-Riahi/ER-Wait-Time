
from fastapi  import APIRouter
from pydantic import BaseModel
from typing   import Any, Dict
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from data_loader   import load_raw
from preprocessing import clean, prepare, apply_pca
from evaluate      import compute_metrics
from backend.model_registry_be import build_model, MODEL_REGISTRY
import mlflow, mlflow.sklearn, joblib, uuid

router    = APIRouter()
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("ER_WaitTime_Prediction")


class TrainRequest(BaseModel):
    model:       str
    hyperparams: Dict[str, Any] = {}
    use_pca:     bool           = False
    pca_n:       int            = 5


@router.post("/")
def train(req: TrainRequest):
    """Train a model from the frontend and log to MLflow"""

    # Load + preprocess data
    df                                              = load_raw()
    df                                              = clean(df)
    X_train, X_test, y_train, y_test, scaler, feats = prepare(df)

    # Apply PCA if requested
    if req.use_pca:
        X_train, X_test, pca = apply_pca(X_train, X_test, req.pca_n)

    # Build model
    model = build_model(req.model, req.hyperparams)

    # Train + evaluate
    with mlflow.start_run(run_name=f"{req.model}_UI"):
        mlflow.log_param("model_type", model.__class__.__name__)
        mlflow.log_param("use_pca",    req.use_pca)
        for k, v in req.hyperparams.items():
            mlflow.log_param(k, v)

        model.fit(X_train, y_train)
        metrics = compute_metrics(model, X_test, y_test)

        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")

    # Save to disk
    path = os.path.join(MODELS_DIR, f"{req.model}_latest.joblib")
    joblib.dump({"model": model, "scaler": scaler, "feature_names": feats}, path)

    exp_id = f"exp_{uuid.uuid4().hex[:6]}"
    return {"exp_id": exp_id, "model": req.model, **metrics}


@router.get("/models")
def list_models():
    """Return available models for the frontend dropdown"""
    return [
        {
            "key":    k,
            "name":   v["name"],
            "family": v["family"],
            "desc":   v["description"],
        }
        for k, v in MODEL_REGISTRY.items()
    ]
