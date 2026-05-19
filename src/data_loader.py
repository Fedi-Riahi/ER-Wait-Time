import pandas as pd
import numpy as np
import os
import io
import sys

# Allow importing config from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import DATASET_PATH, PROCESSED_PATH


def load_raw() -> pd.DataFrame:
    """Load the raw CSV dataset. Falls back to synthetic data if file not found."""
    if not os.path.exists(DATASET_PATH):
        print(f"[DataLoader] '{DATASET_PATH}' not found — generating synthetic data")
        return _generate_synthetic()
    df = pd.read_csv(DATASET_PATH)
    print(f"[DataLoader] Loaded {len(df):,} rows × {len(df.columns)} columns from '{DATASET_PATH}'")
    return df


def load_from_bytes(file_bytes: bytes) -> pd.DataFrame:
    """Load dataset from uploaded file bytes (used by /data/upload endpoint)."""
    try:
        df = pd.read_csv(io.BytesIO(file_bytes))
        print(f"[DataLoader] Loaded uploaded file: {len(df):,} rows")
        return df
    except Exception as e:
        raise ValueError(f"Invalid CSV file: {e}")


def save_processed(df: pd.DataFrame):
    """Save cleaned dataset to the processed folder."""
    os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)
    print(f"[DataLoader] Saved processed data → {PROCESSED_PATH}")


def load_processed() -> pd.DataFrame:
    """Load the cleaned dataset (falls back to raw if processed doesn't exist)."""
    if os.path.exists(PROCESSED_PATH):
        return pd.read_csv(PROCESSED_PATH)
    return load_raw()


def _generate_synthetic(n: int = 3000) -> pd.DataFrame:
    """Fallback synthetic data if no CSV is provided."""
    np.random.seed(42)
    esi      = np.random.choice([1, 2, 3, 4, 5], n, p=[0.02, 0.08, 0.62, 0.20, 0.08])
    docs     = np.random.randint(1, 9, n)
    patients = np.random.randint(5, 50, n)
    age      = np.random.randint(1, 96, n)
    hr       = np.clip(np.random.normal(90,  20, n),  30,  200).astype(int)
    sbp      = np.clip(np.random.normal(120, 20, n),  60,  220).astype(int)
    spo2     = np.clip(np.random.normal(96,   4, n),  70,  100).astype(int)
    occ      = np.clip(0.4 + patients / 60 * 0.5 + np.random.rand(n) * 0.1, 0, 0.95)
    is_wknd  = np.random.randint(0, 2, n)
    is_rush  = np.random.randint(0, 2, n)

    base = np.vectorize({1: 2, 2: 8, 3: 48, 4: 92, 5: 125}.get)(esi)
    wait = np.maximum(
        0,
        base
        + patients * 0.9
        + (40 / (docs + 1))
        + is_rush * 15
        + is_wknd * 8
        + np.random.normal(0, 10, n),
    ).round().astype(int)

    print("[DataLoader] Generated synthetic dataset (3 000 rows)")
    return pd.DataFrame({
        "triage_level_ESI":   esi,
        "doctors_available":  docs,
        "patients_waiting":   patients,
        "age":                age,
        "heart_rate_bpm":     hr,
        "systolic_bp_mmhg":   sbp,
        "spo2_pct":           spo2,
        "occupancy_rate_pct": (occ * 100).round().astype(int),
        "is_weekend":         is_wknd,
        "is_rush_hour":       is_rush,
        "wait_time_min":      wait,
    })
