import mlflow
import pandas as pd
import os
import sys
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import EXPERIMENT_NAME, MODELS_DIR, MLFLOW_TRACKING_URI

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm          import SVR
from sklearn.ensemble     import RandomForestRegressor, AdaBoostRegressor
from xgboost              import XGBRegressor

from data_loader   import load_raw, save_processed
from preprocessing import clean, prepare, apply_pca
from evaluate      import compute_metrics

# ── MLflow setup ──────────────────────────────────────────────────────────────
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

# ── Experiments ───────────────────────────────────────────────────────────────
EXPERIMENTS = [

    # ── Linear Regression ─────────────────────────────────────────────────
    {"name": "LinearRegression_baseline",  "model": LinearRegression(),                                                          "params": {},                                              "use_pca": False},
    {"name": "LinearRegression_with_PCA",  "model": LinearRegression(),                                                          "params": {"pca_components": 5},                           "use_pca": True, "pca_n": 5},

    # ── Ridge ─────────────────────────────────────────────────────────────
    {"name": "Ridge_alpha_0.1",            "model": Ridge(alpha=0.1),                                                            "params": {"alpha": 0.1},                                  "use_pca": False},
    {"name": "Ridge_alpha_1.0",            "model": Ridge(alpha=1.0),                                                            "params": {"alpha": 1.0},                                  "use_pca": False},
    {"name": "Ridge_alpha_10.0",           "model": Ridge(alpha=10.0),                                                           "params": {"alpha": 10.0},                                 "use_pca": False},

    # ── AdaBoost ──────────────────────────────────────────────────────────
    {"name": "AdaBoost_50",                "model": AdaBoostRegressor(n_estimators=50,  random_state=42),                        "params": {"n_estimators": 50,  "learning_rate": 1.0},      "use_pca": False},
    {"name": "AdaBoost_100",               "model": AdaBoostRegressor(n_estimators=100, random_state=42),                        "params": {"n_estimators": 100, "learning_rate": 1.0},      "use_pca": False},
    {"name": "AdaBoost_100_lr0.5",         "model": AdaBoostRegressor(n_estimators=100, learning_rate=0.5, random_state=42),     "params": {"n_estimators": 100, "learning_rate": 0.5},      "use_pca": False},

    # ── XGBoost ───────────────────────────────────────────────────────────
    {"name": "XGBoost_100_lr0.1",          "model": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, verbosity=0),                        "params": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 6}, "use_pca": False},
    {"name": "XGBoost_200_lr0.1",          "model": XGBRegressor(n_estimators=200, learning_rate=0.1, random_state=42, verbosity=0),                        "params": {"n_estimators": 200, "learning_rate": 0.1, "max_depth": 6}, "use_pca": False},
    {"name": "XGBoost_100_lr0.05",         "model": XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42, verbosity=0),          "params": {"n_estimators": 100, "learning_rate": 0.05, "max_depth": 4},"use_pca": False},
    {"name": "XGBoost_200_depth4",         "model": XGBRegressor(n_estimators=200, learning_rate=0.1,  max_depth=4, random_state=42, verbosity=0),          "params": {"n_estimators": 200, "learning_rate": 0.1,  "max_depth": 4},"use_pca": False},

    # ── SVR ───────────────────────────────────────────────────────────────
    {"name": "SVR_rbf_C1",                 "model": SVR(kernel="rbf",    C=1.0),                                                 "params": {"kernel": "rbf",    "C": 1.0},                  "use_pca": False},
    {"name": "SVR_rbf_C5",                 "model": SVR(kernel="rbf",    C=5.0),                                                 "params": {"kernel": "rbf",    "C": 5.0},                  "use_pca": False},
    {"name": "SVR_linear_C1",              "model": SVR(kernel="linear", C=1.0),                                                 "params": {"kernel": "linear", "C": 1.0},                  "use_pca": False},
    {"name": "SVR_rbf_C1_PCA",             "model": SVR(kernel="rbf",    C=1.0),                                                 "params": {"kernel": "rbf",    "C": 1.0, "pca_components": 5}, "use_pca": True, "pca_n": 5},

    # ── Random Forest ─────────────────────────────────────────────────────
    {"name": "RandomForest_50trees",       "model": RandomForestRegressor(n_estimators=50,  random_state=42),                    "params": {"n_estimators": 50,  "max_depth": "None"},       "use_pca": False},
    {"name": "RandomForest_100trees",      "model": RandomForestRegressor(n_estimators=100, random_state=42),                    "params": {"n_estimators": 100, "max_depth": "None"},       "use_pca": False},
    {"name": "RandomForest_200trees",      "model": RandomForestRegressor(n_estimators=200, random_state=42),                    "params": {"n_estimators": 200, "max_depth": "None"},       "use_pca": False},
    {"name": "RandomForest_100trees_d10",  "model": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),      "params": {"n_estimators": 100, "max_depth": 10},           "use_pca": False},
    {"name": "RandomForest_200trees_PCA",  "model": RandomForestRegressor(n_estimators=200, random_state=42),                    "params": {"n_estimators": 200, "pca_components": 5},       "use_pca": True, "pca_n": 5},
]


def run_all_experiments():
    print("=" * 60)
    print("Loading and preparing data...")
    print("=" * 60)

    df = load_raw()
    df = clean(df)
    save_processed(df)

    X_train, X_test, y_train, y_test, scaler, features = prepare(df)

    print(f"\nStarting {len(EXPERIMENTS)} experiments...\n")

    results   = []
    best_r2   = -1
    best_name = None

    for exp in EXPERIMENTS:
        print(f"\n{'─' * 50}")
        print(f"Running: {exp['name']}")
        print(f"{'─' * 50}")

        # Apply PCA if requested
        if exp.get("use_pca"):
            X_tr, X_te, _ = apply_pca(X_train, X_test, n_components=exp.get("pca_n", 5))
        else:
            X_tr, X_te = X_train, X_test

        with mlflow.start_run(run_name=exp["name"]):
            mlflow.log_param("model_type", exp["model"].__class__.__name__)
            mlflow.log_param("use_pca",    exp.get("use_pca", False))
            for key, val in exp["params"].items():
                mlflow.log_param(key, val)

            model = exp["model"]
            model.fit(X_tr, y_train)
            metrics = compute_metrics(model, X_te, y_test)
            mlflow.log_metrics(metrics)

            # ✅ No mlflow.sklearn.log_model() — that's what made it slow.
            # We persist to disk ourselves with joblib below.

        # Save best model to disk
        if metrics["r2"] > best_r2:
            best_r2   = metrics["r2"]
            best_name = exp["name"]
            path = os.path.join(MODELS_DIR, "best_model.joblib")
            joblib.dump(
                {"model": model, "scaler": scaler, "feature_names": features,
                 "experiment": exp["name"], "metrics": metrics},
                path, compress=3,
            )
            print(f"⭐ New best model saved → {path} (R²={metrics['r2']})")

        results.append({
            "Experiment": exp["name"],
            "Model":      exp["model"].__class__.__name__,
            "PCA":        exp.get("use_pca", False),
            "Params":     str(exp["params"]),
            "MAE":        metrics["mae"],
            "MSE":        metrics["mse"],
            "RMSE":       metrics["rmse"],
            "R²":         metrics["r2"],
        })
        print(f"✓ R²={metrics['r2']} | RMSE={metrics['rmse']}")

    _print_comparison_table(results)

    comparison_df = pd.DataFrame(results).sort_values("R²", ascending=False)
    out_csv = os.path.join(MODELS_DIR, "comparison_table.csv")
    comparison_df.to_csv(out_csv, index=False)
    print(f"\n✅ Comparison table saved → {out_csv}")
    print(f"✅ Best model: {best_name} (R²={best_r2})")
    print(f"✅ MLflow UI: run 'mlflow ui' then open http://localhost:5000")

    return results


def _print_comparison_table(results: list):
    print("\n" + "=" * 80)
    print("COMPARISON TABLE — sorted by R² (highest first)")
    print("=" * 80)
    sorted_results = sorted(results, key=lambda x: x["R²"], reverse=True)
    print(f"{'Experiment':<45} {'Model':<25} {'MAE':>7} {'RMSE':>7} {'R²':>7}")
    print("-" * 80)
    for i, r in enumerate(sorted_results):
        marker = " ⭐" if i == 0 else ""
        print(f"{r['Experiment']:<45} {r['Model']:<25} {r['MAE']:>7.3f} {r['RMSE']:>7.3f} {r['R²']:>7.3f}{marker}")
    print("=" * 80)
    print(f"Best: {sorted_results[0]['Experiment']} — R²={sorted_results[0]['R²']}")


if __name__ == "__main__":
    run_all_experiments()
