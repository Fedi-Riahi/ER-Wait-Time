
import pandas as pd
import numpy as np
from sklearn.preprocessing    import StandardScaler, LabelEncoder
from sklearn.model_selection  import train_test_split
from sklearn.decomposition    import PCA
from sklearn.manifold         import TSNE

TARGET = "wait_time_min"

# Columns that are not useful for ML
DROP_COLS = [
    "visit_id", "id_visite", "date", "heure_arrivee",
    "jour_semaine", "day_of_week", "saison", "season",
    "motif_consultation", "chief_complaint",
    "final_disposition", "disposition_finale",
]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicates, fix nulls, remove outliers"""

    before = len(df)
    df = df.drop_duplicates()

    # Fill numeric nulls with median
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
            df = df[df[col].between(mean - 4*std, mean + 4*std)]

    print(f"[Preprocessing] Cleaned: {before:,} → {len(df):,} rows")
    return df.reset_index(drop=True)


def prepare(df: pd.DataFrame):
    """
    Full preprocessing pipeline
    """

    df = df.copy()

    # Drop irrelevant columns
    drop = [c for c in DROP_COLS if c in df.columns]
    df   = df.drop(columns=drop)

    # Separate features and target
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # Encode text columns → numbers
    for col in X.select_dtypes(include="object").columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    feature_names = list(X.columns)

    # Scale all features to mean=0, std=1
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 80/20 train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    print(f"[Preprocessing] Features: {feature_names}")
    print(f"[Preprocessing] Train={len(X_train):,} | Test={len(X_test):,}")

    return X_train, X_test, y_train, y_test, scaler, feature_names


def apply_pca(X_train, X_test, n_components: int = 5):
    """
    Reduce dimensions with PCA
    Keeps the top n components that explain the most variance
    """
    pca        = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca  = pca.transform(X_test)

    explained = pca.explained_variance_ratio_.sum() * 100
    print(f"[PCA] {n_components} components explain {explained:.1f}% of variance")

    return X_train_pca, X_test_pca, pca


def apply_tsne(X, n_components: int = 2):
    """
    t-SNE for 2D visualization only (not for training)
    Returns 2D coordinates for plotting
    """
    tsne   = TSNE(n_components=n_components, random_state=42, perplexity=30)
    X_2d   = tsne.fit_transform(X)
    print(f"[t-SNE] Reduced to {n_components}D for visualization")
    return X_2d
