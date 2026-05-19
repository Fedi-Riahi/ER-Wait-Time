"""
config.py — project root
Single source of truth for all paths and settings.
Both src/ and backend/ import from here.
"""
import os

# ── Project root (the folder this file lives in) ─────────────────────────────
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))

# ── Directory layout ──────────────────────────────────────────────────────────
SRC_DIR    = os.path.join(ROOT_DIR, "src")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
PLOTS_DIR  = os.path.join(ROOT_DIR, "models", "plots")
DATA_DIR   = os.path.join(ROOT_DIR, "data")

# ── Dataset paths ─────────────────────────────────────────────────────────────
# Change DATASET_FILENAME here to switch datasets across the entire project.
DATASET_FILENAME = "er_wait_time_dataset.csv"
DATASET_PATH     = os.path.join(DATA_DIR, "raw", DATASET_FILENAME)
PROCESSED_PATH   = os.path.join(DATA_DIR, "processed", "er_clean.csv")

# ── MLflow ────────────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = f"sqlite:///{os.path.join(ROOT_DIR, 'mlflow.db')}"
EXPERIMENT_NAME     = "ER_WaitTime_Prediction"

# ── API / CORS ────────────────────────────────────────────────────────────────
ALLOWED_ORIGINS = os.environ.get(
    "CORS_ORIGINS", "http://localhost:3000"
).split(",")

# ── Ensure key directories exist ──────────────────────────────────────────────
for _dir in [MODELS_DIR, PLOTS_DIR, os.path.join(DATA_DIR, "raw"), os.path.join(DATA_DIR, "processed")]:
    os.makedirs(_dir, exist_ok=True)
