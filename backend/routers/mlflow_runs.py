from fastapi import APIRouter
import mlflow, os, pandas as pd
from mlflow.tracking import MlflowClient
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import MLFLOW_TRACKING_URI, EXPERIMENT_NAME, MODELS_DIR

router = APIRouter()

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient()


def _get_exp_id() -> str | None:
    exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    return exp.experiment_id if exp else None


def get_runs() -> list[dict]:
    """Return all MLflow runs sorted by R² desc."""
    exp_id = _get_exp_id()
    if not exp_id:
        return []

    runs = client.search_runs(
        experiment_ids=[exp_id],
        order_by=["metrics.r2 DESC"],
    )
    result = []
    for i, run in enumerate(runs):
        result.append(
            {
                "id":      run.info.run_id[:8],
                "run_id":  run.info.run_id,
                "model":   run.data.params.get("model_type", "unknown"),
                "name":    run.info.run_name,
                "date":    run.info.start_time,
                "params":  run.data.params,
                "metrics": {
                    "r2":   round(run.data.metrics.get("r2",   0), 3),
                    "rmse": round(run.data.metrics.get("rmse", 0), 3),
                    "mae":  round(run.data.metrics.get("mae",  0), 3),
                    "mse":  round(run.data.metrics.get("mse",  0), 3),
                },
                "status": "best" if i == 0 else "done",
            }
        )
    return result


@router.get("/runs")
def list_runs():
    return get_runs()


@router.get("/summary")
def get_summary():
    runs = get_runs()
    if not runs:
        return {"total_runs": 0, "best_r2": 0, "best_rmse": 0, "best_model": "none"}
    best = runs[0]
    return {
        "total_runs": len(runs),
        "best_r2":    best["metrics"]["r2"],
        "best_rmse":  best["metrics"]["rmse"],
        "best_model": best["model"],
        "best_name":  best["name"],
    }


@router.get("/comparison")
def get_comparison():
    csv_path = os.path.join(MODELS_DIR, "comparison_table.csv")
    if not os.path.exists(csv_path):
        return []
    df = pd.read_csv(csv_path).sort_values("R²", ascending=False)
    return df.to_dict(orient="records")
