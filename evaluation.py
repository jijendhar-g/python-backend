"""
Evaluation Module
─────────────────
Computes standard regression metrics to assess model quality:
  • MAE  — Mean Absolute Error
  • RMSE — Root Mean Squared Error
  • MAPE — Mean Absolute Percentage Error
  • R²   — Coefficient of Determination
"""

import json
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import METRICS_PATH


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Evaluate predictions against ground truth.

    Parameters
    ----------
    y_true : actual (de-normalised) prices
    y_pred : predicted (de-normalised) prices

    Returns
    -------
    Dictionary with MAE, RMSE, MAPE, and R² values.
    """
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))

    # MAPE — guard against division by zero
    non_zero_mask = y_true != 0
    if non_zero_mask.sum() > 0:
        mape = float(
            np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
        )
    else:
        mape = 0.0

    metrics = {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "mape": round(mape, 4),
        "r2_score": round(r2, 4),
    }

    # Persist to disk so the /model-metrics endpoint can read them
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics
