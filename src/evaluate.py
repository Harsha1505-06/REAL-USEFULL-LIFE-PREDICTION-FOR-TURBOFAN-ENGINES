"""
Evaluation module for Remaining Useful Life (RUL) prediction.
Computes standard regression metrics (RMSE, MAE, R²) and the official
NASA PHM Asymmetric Scoring Function.
"""

from typing import Dict
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def compute_nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the official NASA PHM 2008 Asymmetric Scoring Function.

    Parameters
    ----------
    y_true : np.ndarray
        True ground truth Remaining Useful Life values.
    y_pred : np.ndarray
        Predicted Remaining Useful Life values.

    Returns
    -------
    score : float
        Sum of asymmetric penalties across all test units.

    Mathematical Definition & Viva Rationale:
    Let error d_i = y_pred_i - y_true_i:
    - If d_i < 0 (Early Prediction):
        Penalty = exp(-d_i / 13) - 1
        The engine is serviced before failure. Incurring early maintenance costs operational
        dollars, but maintains aircraft safety.
    - If d_i >= 0 (Late Prediction):
        Penalty = exp(d_i / 10) - 1
        The engine is predicted to last longer than it actually can. The aircraft remains
        in service with a degraded engine, risking catastrophic failure in flight.
    Because failure is catastrophic, late predictions (d_i >= 0) are penalized exponentially
    steeper (denominator 10 vs 13).
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    # Error d = predicted - actual
    diff = y_pred - y_true
    
    penalties = np.where(
        diff < 0,
        np.exp(-diff / 13.0) - 1.0,  # Early prediction penalty
        np.exp(diff / 10.0) - 1.0    # Late prediction penalty (steeper)
    )
    
    return float(np.sum(penalties))


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute RMSE, MAE, R², and NASA Asymmetric Score.

    Returns
    -------
    metrics : dict
        Dictionary of computed metric names and values.
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    nasa_score = compute_nasa_score(y_true, y_pred)
    
    # Early vs Late error distribution
    diff = y_pred - y_true
    n_early = int(np.sum(diff < 0))
    n_late = int(np.sum(diff > 0))
    n_exact = int(np.sum(diff == 0))
    
    return {
        "RMSE": round(rmse, 2),
        "MAE": round(mae, 2),
        "R2": round(r2, 4),
        "NASA_Score": round(nasa_score, 1),
        "Early_Predictions": n_early,
        "Late_Predictions": n_late,
        "Exact_Predictions": n_exact,
    }


if __name__ == "__main__":
    # Quick sanity check
    y_true_mock = np.array([20, 50, 100])
    y_pred_mock = np.array([15, 60, 100])  # Unit 1 early (-5), Unit 2 late (+10), Unit 3 exact (0)
    res = evaluate_predictions(y_true_mock, y_pred_mock)
    print("Sanity Check Metrics:", res)
