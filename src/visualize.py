
# Generates the plots required for the analysis document

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os

from data_loader   import load_raw
from preprocessing import clean, prepare, apply_pca, apply_tsne

PLOTS_DIR = "models/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)


def plot_pca_variance(X_train):
    """
    Plot how much variance each PCA component explains
    """
    from sklearn.decomposition import PCA

    pca = PCA(random_state=42)
    pca.fit(X_train)

    cumulative = np.cumsum(pca.explained_variance_ratio_) * 100

    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(cumulative)+1), cumulative, marker='o', color='#3b82f6')
    plt.axhline(y=90, color='red', linestyle='--', label='90% threshold')
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance (%)")
    plt.title("PCA — Explained Variance per Component")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "pca_variance.png")
    plt.savefig(path)
    plt.close()
    print(f"[Visualize] Saved → {path}")


def plot_tsne(X_train, y_train):
    """
    t-SNE: visualize the data in 2D to see if there are natural clusters
    Color = wait time (low=green, high=red)
    """
    print("[Visualize] Running t-SNE (this takes ~30 seconds)...")

    # Use a sample to keep it fast
    n_sample = min(1000, len(X_train))
    idx      = np.random.choice(len(X_train), n_sample, replace=False)
    X_sample = X_train[idx]
    y_sample = np.array(y_train)[idx]

    X_2d = apply_tsne(X_sample, n_components=2)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        X_2d[:, 0], X_2d[:, 1],
        c=y_sample, cmap='RdYlGn_r',
        alpha=0.6, s=20
    )
    plt.colorbar(scatter, label="Wait Time (min)")
    plt.title("t-SNE — ER Data in 2D (color = wait time)")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "tsne.png")
    plt.savefig(path)
    plt.close()
    print(f"[Visualize] Saved → {path}")


def plot_model_comparison(csv_path: str = "models/comparison_table.csv"):
    """
    Bar chart comparing R² of all models
    """
    if not os.path.exists(csv_path):
        print("[Visualize] Run train.py first to generate comparison_table.csv")
        return

    df = pd.read_csv(csv_path).sort_values("R²", ascending=False)

    plt.figure(figsize=(12, 5))
    colors = ["#3b82f6" if i == 0 else "#64748b" for i in range(len(df))]
    bars   = plt.barh(df["Experiment"], df["R²"], color=colors)
    plt.xlabel("R² Score")
    plt.title("Model Comparison — R² Score (higher = better)")
    plt.axvline(x=0.8, color='red', linestyle='--', label='Target R²=0.8')
    plt.legend()
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "model_comparison_r2.png")
    plt.savefig(path)
    plt.close()
    print(f"[Visualize] Saved → {path}")

    # RMSE comparison
    plt.figure(figsize=(12, 5))
    plt.barh(df["Experiment"], df["RMSE"], color="#ef4444")
    plt.xlabel("RMSE (minutes)")
    plt.title("Model Comparison — RMSE (lower = better)")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "model_comparison_rmse.png")
    plt.savefig(path)
    plt.close()
    print(f"[Visualize] Saved → {path}")


if __name__ == "__main__":
    df                                              = load_raw()
    df                                              = clean(df)
    X_train, X_test, y_train, y_test, scaler, feats = prepare(df)

    plot_pca_variance(X_train)
    plot_tsne(X_train, y_train)
    plot_model_comparison()

    print("\n✅ All plots saved to models/plots/")
