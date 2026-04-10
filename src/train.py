
import mlflow
import mlflow.sklearn
import pandas as pd
import os
import joblib

from sklearn.linear_model  import LinearRegression, Ridge
from sklearn.svm           import SVR
from sklearn.ensemble      import RandomForestRegressor

from data_loader    import load_raw, save_processed
from preprocessing  import clean, prepare, apply_pca
from evaluate       import compute_metrics

# ── Config ────────────────────────────────────────────────────────────────────
EXPERIMENT_NAME = "ER_WaitTime_Prediction"
MODELS_DIR      = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# ── MLflow setup ──────────────────────────────────────────────────────────────
# All runs will appear under this experiment in the MLflow UI
mlflow.set_experiment(EXPERIMENT_NAME)


# ── Define all experiments to run ─────────────────────────────────────────────
# Each dict = one MLflow run
# This is how you "test different hyperparameter values" as the task asks

EXPERIMENTS = [

    # ── Linear Regression (baseline) ──────────────────────────────────────
    {
        "name":      "LinearRegression_baseline",
        "model":     LinearRegression(),
        "params":    {},
        "use_pca":   False,
    },
    {
        "name":      "LinearRegression_with_PCA",
        "model":     LinearRegression(),
        "params":    {"pca_components": 5},
        "use_pca":   True,
        "pca_n":     5,
    },

    # ── Ridge Regression (regularized linear) ─────────────────────────────
    {
        "name":      "Ridge_alpha_0.1",
        "model":     Ridge(alpha=0.1),
        "params":    {"alpha": 0.1},
        "use_pca":   False,
    },
    {
        "name":      "Ridge_alpha_1.0",
        "model":     Ridge(alpha=1.0),
        "params":    {"alpha": 1.0},
        "use_pca":   False,
    },
    {
        "name":      "Ridge_alpha_10.0",
        "model":     Ridge(alpha=10.0),
        "params":    {"alpha": 10.0},
        "use_pca":   False,
    },

    # ── SVR ───────────────────────────────────────────────────────────────
    {
        "name":      "SVR_rbf_C1",
        "model":     SVR(kernel="rbf", C=1.0),
        "params":    {"kernel": "rbf", "C": 1.0},
        "use_pca":   False,
    },
    {
        "name":      "SVR_rbf_C5",
        "model":     SVR(kernel="rbf", C=5.0),
        "params":    {"kernel": "rbf", "C": 5.0},
        "use_pca":   False,
    },
    {
        "name":      "SVR_linear_C1",
        "model":     SVR(kernel="linear", C=1.0),
        "params":    {"kernel": "linear", "C": 1.0},
        "use_pca":   False,
    },
    {
        "name":      "SVR_rbf_C1_PCA",
        "model":     SVR(kernel="rbf", C=1.0),
        "params":    {"kernel": "rbf", "C": 1.0, "pca_components": 5},
        "use_pca":   True,
        "pca_n":     5,
    },

    # ── Random Forest ─────────────────────────────────────────────────────
    {
        "name":      "RandomForest_50trees",
        "model":     RandomForestRegressor(n_estimators=50, random_state=42),
        "params":    {"n_estimators": 50, "max_depth": "None"},
        "use_pca":   False,
    },
    {
        "name":      "RandomForest_100trees",
        "model":     RandomForestRegressor(n_estimators=100, random_state=42),
        "params":    {"n_estimators": 100, "max_depth": "None"},
        "use_pca":   False,
    },
    {
        "name":      "RandomForest_200trees",
        "model":     RandomForestRegressor(n_estimators=200, random_state=42),
        "params":    {"n_estimators": 200, "max_depth": "None"},
        "use_pca":   False,
    },
    {
        "name":      "RandomForest_100trees_depth10",
        "model":     RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        "params":    {"n_estimators": 100, "max_depth": 10},
        "use_pca":   False,
    },
    {
        "name":      "RandomForest_200trees_PCA",
        "model":     RandomForestRegressor(n_estimators=200, random_state=42),
        "params":    {"n_estimators": 200, "pca_components": 5},
        "use_pca":   True,
        "pca_n":     5,
    },
]


def run_all_experiments():
    """
    Train every model in EXPERIMENTS list.
    Each one becomes a separate MLflow run.
    """

    # ── Load and prepare data once ────────────────────────────────────────────
    print("=" * 60)
    print("Loading and preparing data...")
    print("=" * 60)

    df = load_raw()
    df = clean(df)
    save_processed(df)

    X_train, X_test, y_train, y_test, scaler, features = prepare(df)

    print(f"\nStarting {len(EXPERIMENTS)} experiments...\n")

    results = []   # collect all results for comparison table at the end

    for exp in EXPERIMENTS:
        print(f"\n{'─' * 50}")
        print(f"Running: {exp['name']}")
        print(f"{'─' * 50}")

        # ── Apply PCA if requested ────────────────────────────────────────────
        if exp.get("use_pca"):
            n = exp.get("pca_n", 5)
            X_tr, X_te, pca = apply_pca(X_train, X_test, n_components=n)
        else:
            X_tr, X_te = X_train, X_test

        # ── Start MLflow run ──────────────────────────────────────────────────
        with mlflow.start_run(run_name=exp["name"]):

            # 1. Log parameters — what settings we used
            mlflow.log_param("model_type", exp["model"].__class__.__name__)
            mlflow.log_param("use_pca",    exp.get("use_pca", False))
            for key, val in exp["params"].items():
                mlflow.log_param(key, val)

            # 2. Train the model
            model = exp["model"]
            model.fit(X_tr, y_train)

            # 3. Evaluate — compute metrics on test set
            metrics = compute_metrics(model, X_te, y_test)

            # 4. Log metrics to MLflow
            mlflow.log_metric("mae",  metrics["mae"])
            mlflow.log_metric("mse",  metrics["mse"])
            mlflow.log_metric("rmse", metrics["rmse"])
            mlflow.log_metric("r2",   metrics["r2"])

            # 5. Save model artifact inside MLflow
            mlflow.sklearn.log_model(model, "model")

            # 6. Also save best model to disk separately
            _save_if_best(model, scaler, features, exp["name"], metrics)

            # 7. Collect result for comparison table
            results.append({
                "Experiment":  exp["name"],
                "Model":       exp["model"].__class__.__name__,
                "PCA":         exp.get("use_pca", False),
                "Params":      str(exp["params"]),
                "MAE":         metrics["mae"],
                "MSE":         metrics["mse"],
                "RMSE":        metrics["rmse"],
                "R²":          metrics["r2"],
            })

            print(f"✓ Logged to MLflow — R²={metrics['r2']} | RMSE={metrics['rmse']}")

    # ── Print comparison table
    _print_comparison_table(results)

    # ── Save comparison table to CSV
    comparison_df = pd.DataFrame(results)
    comparison_df = comparison_df.sort_values("R²", ascending=False)
    comparison_df.to_csv("models/comparison_table.csv", index=False)
    print("\n✅ Comparison table saved → models/comparison_table.csv")
    print(f"✅ MLflow UI: run 'mlflow ui' then open http://localhost:5000")

    return results


# ── Helpers

_best_r2    = -1
_best_name  = None

def _save_if_best(model, scaler, features, name, metrics):
    """Save this model to disk if it has the best R² so far"""
    global _best_r2, _best_name

    if metrics["r2"] > _best_r2:
        _best_r2   = metrics["r2"]
        _best_name = name

        path = os.path.join(MODELS_DIR, "best_model.joblib")
        joblib.dump({
            "model":         model,
            "scaler":        scaler,
            "feature_names": features,
            "experiment":    name,
            "metrics":       metrics,
        }, path)
        print(f"⭐ New best model saved → {path} (R²={metrics['r2']})")


def _print_comparison_table(results: list):
    """Print a formatted comparison table to the terminal"""
    print("\n")
    print("=" * 80)
    print("COMPARISON TABLE — sorted by R² (highest first)")
    print("=" * 80)

    sorted_results = sorted(results, key=lambda x: x["R²"], reverse=True)

    # Header
    print(f"{'Experiment':<45} {'Model':<25} {'MAE':>7} {'RMSE':>7} {'R²':>7}")
    print("-" * 80)

    for r in sorted_results:
        marker = " ⭐" if r == sorted_results[0] else ""
        print(
            f"{r['Experiment']:<45} "
            f"{r['Model']:<25} "
            f"{r['MAE']:>7.3f} "
            f"{r['RMSE']:>7.3f} "
            f"{r['R²']:>7.3f}"
            f"{marker}"
        )

    print("=" * 80)
    print(f"Best model: {sorted_results[0]['Experiment']} — R²={sorted_results[0]['R²']}")


# ── Entry point

if __name__ == "__main__":
    run_all_experiments()
