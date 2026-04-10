
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def compute_metrics(model, X_test, y_test) -> dict:
    """
    Compute all regression metrics required by the task:
      - MAE   (Mean Absolute Error)
      - MSE   (Mean Squared Error)
      - RMSE  (Root Mean Squared Error)
      - R²    (Coefficient of Determination)
    """
    predictions = model.predict(X_test)
    y_arr       = np.array(y_test)

    mae  = mean_absolute_error(y_arr, predictions)
    mse  = mean_squared_error(y_arr, predictions)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_arr, predictions)

    # Clamp R² between 0 and 1 for display
    r2 = max(0.0, min(1.0, r2))

    metrics = {
        "mae":  round(mae,  3),
        "mse":  round(mse,  3),
        "rmse": round(rmse, 3),
        "r2":   round(r2,   3),
    }

    print(f"[Evaluate] MAE={metrics['mae']} | MSE={metrics['mse']} "
          f"| RMSE={metrics['rmse']} | R²={metrics['r2']}")

    return metrics
