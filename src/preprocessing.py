import pandas as pd
import numpy as np
from sklearn.preprocessing   import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition   import PCA
from sklearn.manifold        import TSNE

TARGET = "wait_time_min"

# ── Columns to drop before training ──────────────────────────────────────────
# care_duration_min and final_disposition are LEAKAGE — unknown at arrival time
DROP_COLS = [
    "visit_id",          "id_visite",
    "date",              "heure_arrivee",
    "jour_semaine",      "day_of_week",
    "saison",            "season",
    "final_disposition", "disposition_finale",
    "care_duration_min",                        # ← leakage: unknown at arrival
]

# ── Vital sign columns that benefit from triage-grouped imputation ────────────
VITALS = [
    "heart_rate_bpm",
    "systolic_bp_mmhg",
    "diastolic_bp_mmhg",
    "spo2_pct",
    "temperature_C",
]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicates, fix nulls, remove outliers."""
    before = len(df)
    df = df.drop_duplicates()

    # num_comorbidities nulls are almost certainly 0, not median
    if "num_comorbidities" in df.columns:
        df["num_comorbidities"] = df["num_comorbidities"].fillna(0)

    # Vitals: impute within triage group (ESI-1 normals ≠ ESI-5 normals)
    triage_col = next(
        (c for c in ["triage_level_ESI", "niveau_triage"] if c in df.columns), None
    )
    for col in VITALS:
        if col not in df.columns:
            continue
        if triage_col:
            df[col] = df.groupby(triage_col)[col].transform(
                lambda x: x.fillna(x.median())
            )
        df[col] = df[col].fillna(df[col].median())  # global fallback

    # Fill remaining numeric nulls with median
    for col in df.select_dtypes(include="number").columns:
        df[col] = df[col].fillna(df[col].median())

    # Fill text nulls with mode
    for col in df.select_dtypes(include="object").columns:
        mode = df[col].mode()
        if len(mode) > 0:
            df[col] = df[col].fillna(mode[0])

    # Remove extreme outliers (±4 std) — skip the target column
    for col in df.select_dtypes(include="number").columns:
        if col == TARGET:
            continue
        mean, std = df[col].mean(), df[col].std()
        if std > 0:
            df = df[df[col].between(mean - 4 * std, mean + 4 * std)]

    print(f"[Preprocessing] Cleaned: {before:,} → {len(df):,} rows")
    return df.reset_index(drop=True)


def prepare(df: pd.DataFrame):
    """Full preprocessing pipeline — returns train/test splits ready for sklearn."""
    df = df.copy()

    # ── Drop irrelevant / leakage columns ────────────────────────────────────
    drop = [c for c in DROP_COLS if c in df.columns]
    if drop:
        print(f"[Preprocessing] Dropping: {drop}")
    df = df.drop(columns=drop)

    # ── Separate features and target ─────────────────────────────────────────
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # ── chief_complaint: target-encode instead of dropping ───────────────────
    complaint_col = next(
        (c for c in ["chief_complaint", "motif_consultation"] if c in X.columns), None
    )
    if complaint_col:
        complaint_means = df.groupby(complaint_col)[TARGET].mean()
        X[complaint_col] = X[complaint_col].map(complaint_means).fillna(y.mean())
        print(f"[Preprocessing] Target-encoded '{complaint_col}' "
              f"({df[complaint_col].nunique()} categories)")

    # ── sex: binary encode (M=1, F=0) ────────────────────────────────────────
    sex_col = next((c for c in ["sex", "sexe"] if c in X.columns), None)
    if sex_col:
        X[sex_col] = X[sex_col].map({"M": 1, "F": 0, "m": 1, "f": 0}).fillna(0)

    # ── arrival_mode: one-hot encode (genuinely unordered categories) ─────────
    arrival_col = next(
        (c for c in ["arrival_mode", "mode_arrivee"] if c in X.columns), None
    )
    if arrival_col:
        dummies = pd.get_dummies(X[arrival_col], prefix="arrival", drop_first=True)
        X = pd.concat([X.drop(columns=[arrival_col]), dummies], axis=1)
        print(f"[Preprocessing] One-hot encoded '{arrival_col}' "
              f"→ {list(dummies.columns)}")

    # Engineered features 
    if "patients_waiting" in X.columns and "doctors_available" in X.columns:
        X["pressure_index"] = X["patients_waiting"] / (X["doctors_available"] + 1)

    if "patients_waiting" in X.columns and "available_beds" in X.columns:
        X["bed_shortage"] = X["patients_waiting"] - X["available_beds"]

    if "triage_level_ESI" in X.columns and "patient_doctor_ratio" in X.columns:
        X["acuity_pressure"] = X["triage_level_ESI"] * X["patient_doctor_ratio"]

    # ── Label-encode any remaining object columns ─────────────────────────────
    for col in X.select_dtypes(include="object").columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    # ── Ensure all columns are numeric (bool → int for get_dummies output) ────
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    feature_names = list(X.columns)

    # ── Scale to mean=0, std=1 ────────────────────────────────────────────────
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── 80/20 train-test split ────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    print(f"[Preprocessing] Features ({len(feature_names)}): {feature_names}")
    print(f"[Preprocessing] Train={len(X_train):,} | Test={len(X_test):,}")
    return X_train, X_test, y_train, y_test, scaler, feature_names


def apply_pca(X_train, X_test, n_components: int = 5):
    """Reduce dimensions with PCA — keeps top n components by explained variance."""
    pca         = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca  = pca.transform(X_test)
    explained   = pca.explained_variance_ratio_.sum() * 100
    print(f"[PCA] {n_components} components explain {explained:.1f}% of variance")
    return X_train_pca, X_test_pca, pca


def apply_tsne(X, n_components: int = 2):
    """t-SNE for 2D visualization only — do not use for training."""
    tsne = TSNE(n_components=n_components, random_state=42, perplexity=30)
    X_2d = tsne.fit_transform(X)
    print(f"[t-SNE] Reduced to {n_components}D for visualization")
    return X_2d
