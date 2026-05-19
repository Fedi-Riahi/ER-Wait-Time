from sklearn.linear_model   import Ridge
from sklearn.svm             import SVR
from sklearn.ensemble        import RandomForestRegressor, AdaBoostRegressor
from sklearn.neighbors       import KNeighborsRegressor
from sklearn.neural_network  import MLPRegressor
from xgboost                 import XGBRegressor
import inspect

MODEL_REGISTRY = {
    "adaboost": {
        "name":        "AdaBoost",
        "family":      "Ensemble",
        "class":       AdaBoostRegressor,
        "description": "Boosting method that combines weak learners sequentially. Sensitive to outliers.",
        "defaults": {
            "n_estimators":  100,
            "learning_rate": 1.0,
            "loss":          "linear",
            "random_state":  42,
        },
    },
    "xgboost": {
        "name":        "XGBoost",
        "family":      "Gradient Boosting",
        "class":       XGBRegressor,
        "description": "Extreme Gradient Boosting. Often the best performer on tabular data.",
        "defaults": {
            "n_estimators":     100,
            "learning_rate":    0.1,
            "max_depth":        6,
            "subsample":        1.0,
            "colsample_bytree": 1.0,
            "gamma":            0,
            "random_state":     42,
            "verbosity":        0,
        },
    },
    "linear_reg": {
        "name":        "Linear Regression",
        "family":      "Linear",
        "class":       Ridge,
        "description": "Simple linear model. Fast and interpretable.",
        "defaults": {
            "alpha":  1.0,
            "solver": "auto",
        },
    },
    "svr": {
        "name":        "SVR",
        "family":      "Kernel",
        "class":       SVR,
        "description": "Support Vector Regression. Works well with scaled data.",
        "defaults": {
            "kernel":  "rbf",
            "C":       1.0,
            "gamma":   "scale",
            "epsilon": 0.1,
        },
    },
    "random_forest": {
        "name":        "Random Forest",
        "family":      "Ensemble",
        "class":       RandomForestRegressor,
        "description": "Best performer on this dataset. Handles non-linearity well.",
        "defaults": {
            "n_estimators":      100,
            "max_depth":         10,
            "min_samples_split": 2,
            "min_samples_leaf":  1,
            "max_features":      "sqrt",
            "bootstrap":         "true",
            "random_state":      42,
        },
    },
    "knn": {
        "name":        "KNN",
        "family":      "Instance-based",
        "class":       KNeighborsRegressor,
        "description": "Predicts based on nearest neighbors.",
        "defaults": {
            "n_neighbors": 5,
            "weights":     "uniform",
            "metric":      "minkowski",
        },
    },
    "neural_net": {
        "name":        "Neural Network",
        "family":      "Deep Learning",
        "class":       MLPRegressor,
        "description": "Multi-layer perceptron. Powerful but slower.",
        "defaults": {
            "learning_rate_init": 0.001,
            "hidden_layer_sizes": "(128,64)",
            "activation":         "relu",
            "solver":             "adam",
            "batch_size":         32,
            "max_iter":           300,
            "alpha":              0.0001,
            "random_state":       42,
        },
    },
}

# ── Params never shown in the UI ──────────────────────────────────────────────
_HIDDEN = {"random_state", "verbosity", "n_jobs", "warm_start", "oob_score"}

# ── Categorical params and their allowed options ──────────────────────────────
_CATEGORICALS = {
    "loss":               ["linear", "square", "exponential"],
    "kernel":             ["rbf", "linear", "poly", "sigmoid"],
    "weights":            ["uniform", "distance"],
    "metric":             ["euclidean", "manhattan", "minkowski"],
    "max_features":       ["sqrt", "log2"],
    "bootstrap":          ["true", "false"],
    "solver":             ["auto", "svd", "cholesky", "lsqr", "sag", "saga", "adam", "sgd"],
    "activation":         ["relu", "tanh", "logistic"],
    "hidden_layer_sizes": ["(64,)", "(128,)", "(128,64)", "(256,128)", "(256,128,64)"],
}

# ── Range bounds for numeric params ──────────────────────────────────────────
_RANGES = {
    "n_estimators":       {"min": 10,      "max": 1000,  "step": 10},
    "learning_rate":      {"min": 0.01,    "max": 2.0,   "step": 0.01},
    "learning_rate_init": {"min": 0.0001,  "max": 0.1,   "step": 0.0001},
    "max_depth":          {"min": 1,       "max": 100,    "step": 1},
    "min_samples_split":  {"min": 2,       "max": 20,    "step": 1},
    "min_samples_leaf":   {"min": 1,       "max": 20,    "step": 1},
    "n_neighbors":        {"min": 1,       "max": 50,    "step": 1},
    "C":                  {"min": 0.1,     "max": 100,   "step": 0.1},
    "epsilon":            {"min": 0.01,    "max": 1.0,   "step": 0.01},
    "alpha":              {"min": 0.0001,  "max": 100,   "step": 0.0001},
    "subsample":          {"min": 0.5,     "max": 1.0,   "step": 0.05},
    "colsample_bytree":   {"min": 0.5,     "max": 1.0,   "step": 0.05},
    "gamma":              {"min": 0,       "max": 10,    "step": 0.01},
    "batch_size":         {"min": 16,      "max": 256,   "step": 16},
    "max_iter":           {"min": 100,     "max": 1000,  "step": 50},
}

# ── Human-readable labels ─────────────────────────────────────────────────────
_LABELS = {
    "n_estimators":       "Estimators",
    "learning_rate":      "Learning Rate",
    "learning_rate_init": "Learning Rate",
    "max_depth":          "Max Depth",
    "min_samples_split":  "Min Split",
    "min_samples_leaf":   "Min Leaf",
    "n_neighbors":        "Neighbors",
    "C":                  "C (Regularization)",
    "epsilon":            "Epsilon",
    "alpha":              "Alpha",
    "subsample":          "Subsample",
    "colsample_bytree":   "Feature Sample",
    "gamma":              "Gamma",
    "batch_size":         "Batch Size",
    "max_iter":           "Epochs",
    "kernel":             "Kernel",
    "loss":               "Loss",
    "weights":            "Weights",
    "metric":             "Metric",
    "max_features":       "Max Features",
    "bootstrap":          "Bootstrap",
    "solver":             "Solver / Optimizer",
    "activation":         "Activation",
    "hidden_layer_sizes": "Architecture",
}


def _infer_hyperparam(key: str, default) -> dict | None:
    if key in _HIDDEN:
        return None

    label = _LABELS.get(key, key.replace("_", " ").title())

    if key in _CATEGORICALS:
        options = _CATEGORICALS[key]
        if isinstance(default, (tuple, list)):
            str_default = str(tuple(default)).replace(" ", "")
        else:
            str_default = str(default).lower()
        safe_default = str_default if str_default in options else options[0]
        return {"key": key, "label": label, "type": "select",
                "options": options, "default": safe_default}

    if key in _RANGES and isinstance(default, (int, float)):
        r = _RANGES[key]
        return {"key": key, "label": label, "type": "range",
                "min": r["min"], "max": r["max"], "step": r["step"],
                "default": default}

    return None


def get_hyperparams(key: str) -> list[dict]:
    entry  = MODEL_REGISTRY[key]
    schema = []
    for param_key, default in entry["defaults"].items():
        descriptor = _infer_hyperparam(param_key, default)
        if descriptor:
            schema.append(descriptor)
    return schema


def build_model(key: str, params: dict):
    entry      = MODEL_REGISTRY[key]
    final      = {**entry["defaults"], **params}
    ModelClass = entry["class"]
    valid      = inspect.signature(ModelClass.__init__).parameters.keys()
    clean      = {k: v for k, v in final.items() if k in valid}
    return ModelClass(**clean)
