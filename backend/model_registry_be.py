
from sklearn.linear_model   import LinearRegression, Ridge
from sklearn.svm             import SVR
from sklearn.ensemble        import RandomForestRegressor
from sklearn.neighbors       import KNeighborsRegressor
from sklearn.neural_network  import MLPRegressor
from sklearn.ensemble import AdaBoostRegressor
from xgboost          import XGBRegressor
MODEL_REGISTRY = {
    "adaboost": {
    "name":        "AdaBoost",
    "family":      "Ensemble",
    "class":       AdaBoostRegressor,
    "description": "Boosting method that combines weak learners sequentially. Sensitive to outliers.",
    "defaults":    {"n_estimators": 100, "learning_rate": 1.0, "random_state": 42},
    },
    "xgboost": {
        "name":        "XGBoost",
        "family":      "Gradient Boosting",
        "class":       XGBRegressor,
        "description": "Extreme Gradient Boosting. Often the best performer on tabular data.",
        "defaults":    {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 6, "random_state": 42, "verbosity": 0},
    },
    "linear_reg": {
        "name":        "Linear Regression",
        "family":      "Linear",
        "class":       Ridge,
        "description": "Simple linear model. Fast and interpretable.",
        "defaults":    {"alpha": 1.0},
    },
    "svr": {
        "name":        "SVR",
        "family":      "Kernel",
        "class":       SVR,
        "description": "Support Vector Regression. Works well with scaled data.",
        "defaults":    {"kernel": "rbf", "C": 1.0},
    },
    "random_forest": {
        "name":        "Random Forest",
        "family":      "Ensemble",
        "class":       RandomForestRegressor,
        "description": "Best performer on this dataset. Handles non-linearity well.",
        "defaults":    {"n_estimators": 100, "random_state": 42},
    },
    "knn": {
        "name":        "KNN",
        "family":      "Instance-based",
        "class":       KNeighborsRegressor,
        "description": "Predicts based on nearest neighbors.",
        "defaults":    {"n_neighbors": 5},
    },
    "neural_net": {
        "name":        "Neural Network",
        "family":      "Deep Learning",
        "class":       MLPRegressor,
        "description": "Multi-layer perceptron. Powerful but slower.",
        "defaults":    {"hidden_layer_sizes": (128, 64), "max_iter": 300, "random_state": 42},
    },
}


def build_model(key: str, params: dict):
    entry       = MODEL_REGISTRY[key]
    final       = {**entry["defaults"], **params}
    ModelClass  = entry["class"]

    import inspect
    valid = inspect.signature(ModelClass.__init__).parameters.keys()
    clean = {k: v for k, v in final.items() if k in valid}

    return ModelClass(**clean)
