from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Any, Dict
import sys, os, uuid, time, joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from data_loader        import load_raw
from preprocessing      import clean, prepare, apply_pca
from evaluate           import compute_metrics
from model_registry_be  import build_model, MODEL_REGISTRY, get_hyperparams
from config             import MLFLOW_TRACKING_URI, EXPERIMENT_NAME, MODELS_DIR
import mlflow

router = APIRouter()

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)


class TrainRequest(BaseModel):
    model:       str
    hyperparams: Dict[str, Any] = {}
    use_pca:     bool           = False
    pca_n:       int            = 5


@router.post("/")
def train(req: TrainRequest):
    if req.model not in MODEL_REGISTRY:
        raise HTTPException(status_code=400, detail=f"Unknown model key: '{req.model}'")

    t0 = time.perf_counter()

    # ── Load & preprocess ────────────────────────────────────────────────────
    df = load_raw()
    df = clean(df)
    X_train, X_test, y_train, y_test, scaler, feats = prepare(df)

    if req.use_pca:
        X_train, X_test, _ = apply_pca(X_train, X_test, req.pca_n)

    # ── Train ────────────────────────────────────────────────────────────────
    model = build_model(req.model, req.hyperparams)
    model.fit(X_train, y_train)

    # ── Evaluate ─────────────────────────────────────────────────────────────
    metrics = compute_metrics(model, X_test, y_test)

    # ── Log to MLflow (params + metrics only — NO log_model; that's what's slow) ──
    with mlflow.start_run(run_name=f"{req.model}_UI"):
        mlflow.log_param("model_type", model.__class__.__name__)
        mlflow.log_param("use_pca",    req.use_pca)
        mlflow.log_param("n_features", len(feats))
        for k, v in req.hyperparams.items():
            mlflow.log_param(k, v)
        mlflow.log_metrics(metrics)
        # ✅ We intentionally skip mlflow.sklearn.log_model() here.
        # That call serialises and uploads the full model artifact to MLflow
        # storage, which adds 1–3 s per run with no benefit since we already
        # save the model to disk via joblib below.

    # ── Persist to disk ──────────────────────────────────────────────────────
    path = os.path.join(MODELS_DIR, f"{req.model}_latest.joblib")
    joblib.dump(
        {"model": model, "scaler": scaler, "feature_names": feats},
        path,
        compress=3,  # slight compression, negligible speed cost
    )

    elapsed = round(time.perf_counter() - t0, 2)
    return {
        "exp_id":      f"exp_{uuid.uuid4().hex[:6]}",
        "model":       req.model,
        "train_time_s": elapsed,
        **metrics,
    }


@router.get("/models")
def list_models():
    return [
        {
            "key":        k,
            "name":       v["name"],
            "family":     v["family"],
            "desc":       v["description"],
            "hyperparams": get_hyperparams(k),   # ← derived, never hardcoded
        }
        for k, v in MODEL_REGISTRY.items()
    ]
