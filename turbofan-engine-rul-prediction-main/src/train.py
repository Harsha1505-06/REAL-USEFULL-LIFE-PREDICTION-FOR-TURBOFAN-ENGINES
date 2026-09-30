"""
Training and evaluation pipeline for classical ML models:
1. Linear Regression (Baseline)
2. Random Forest Regressor
3. XGBoost Regressor

Evaluates on both validation engines (grouped split) and official test engines.
Generates feature importance plots and test comparison charts.
"""

from pathlib import Path
import sys
from typing import Dict, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Ensure parent directory is on sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.config import PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR, REPORTS_DIR, RANDOM_SEED
from src.models import get_linear_regression, get_random_forest, get_xgboost
from src.evaluate import evaluate_predictions


def load_classical_data(subset: str = "FD001"):
    """Load pre-engineered tabular training, validation, and test datasets."""
    X_train = pd.read_parquet(PROCESSED_DATA_DIR / f"{subset}_X_train_tabular.parquet")
    y_train = pd.read_parquet(PROCESSED_DATA_DIR / f"{subset}_y_train_tabular.parquet")["RUL"].values
    
    X_val = pd.read_parquet(PROCESSED_DATA_DIR / f"{subset}_X_val_tabular.parquet")
    y_val = pd.read_parquet(PROCESSED_DATA_DIR / f"{subset}_y_val_tabular.parquet")["RUL"].values
    
    X_test = pd.read_parquet(PROCESSED_DATA_DIR / f"{subset}_X_test_tabular.parquet")
    y_test = np.load(PROCESSED_DATA_DIR / f"{subset}_y_test_tabular.npy")
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def plot_feature_importances(rf_model, xgb_model, feature_names, output_path: Path):
    """
    Generate side-by-side bar plots of top feature importances for Random Forest and XGBoost.
    """
    rf_imp = pd.Series(rf_model.feature_importances_, index=feature_names).sort_values(ascending=False).head(12)
    xgb_imp = pd.Series(xgb_model.feature_importances_, index=feature_names).sort_values(ascending=False).head(12)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    sns.barplot(x=rf_imp.values, y=rf_imp.index, ax=axes[0], palette="Blues_r")
    axes[0].set_title("Top 12 Features: Random Forest (MDI / Impurity)", fontsize=11, fontweight="semibold")
    axes[0].set_xlabel("Relative Importance")
    
    sns.barplot(x=xgb_imp.values, y=xgb_imp.index, ax=axes[1], palette="Greens_r")
    axes[1].set_title("Top 12 Features: XGBoost (Gain)", fontsize=11, fontweight="semibold")
    axes[1].set_xlabel("Relative Importance")
    
    plt.suptitle("Feature Importance Ranking on Engineered Rolling Telemetry (FD001)", fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved feature importance plot to {output_path}")


def plot_predictions_vs_actual(y_test, predictions_dict: Dict[str, np.ndarray], output_path: Path):
    """
    Plot predicted RUL vs ground truth RUL across all 100 test engines, sorted by actual RUL.
    """
    # Sort test engines by increasing ground truth RUL for visual clarity
    sort_idx = np.argsort(y_test)
    y_sorted = y_test[sort_idx]
    
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(np.arange(len(y_test)), y_sorted, label="Ground Truth RUL", color="black", linewidth=2.2, linestyle="--")
    
    colors = {"Linear Regression": "#7f8c8d", "Random Forest": "#2980b9", "XGBoost": "#27ae60"}
    for name, preds in predictions_dict.items():
        preds_sorted = preds[sort_idx]
        ax.plot(np.arange(len(y_test)), preds_sorted, label=name, color=colors.get(name, "orange"), alpha=0.8, linewidth=1.5)
        
    ax.set_title("Test Engine RUL Predictions vs Ground Truth (Sorted by Actual RUL)", fontsize=12, fontweight="semibold")
    ax.set_xlabel("Test Engines (Ranked by Ascending Lifespan)")
    ax.set_ylabel("Remaining Useful Life (Cycles)")
    ax.legend(frameon=True, loc="upper left")
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved prediction comparison plot to {output_path}")


def train_classical_models(subset: str = "FD001"):
    """Train, validate, test, and persist all classical models."""
    print(f"\n==========================================")
    print(f" Training Classical ML Models on {subset}")
    print(f"==========================================")
    
    X_train, y_train, X_val, y_val, X_test, y_test = load_classical_data(subset)
    feature_names = X_train.columns.tolist()
    
    models = {
        "Linear Regression": get_linear_regression(),
        "Random Forest": get_random_forest(n_estimators=100, max_depth=12, random_state=RANDOM_SEED),
        "XGBoost": get_xgboost(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=RANDOM_SEED)
    }
    
    results = []
    test_preds = {"Unit": np.arange(1, len(y_test) + 1), "Ground_Truth_RUL": y_test}
    preds_dict_for_plot = {}

    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        model.fit(X_train, y_train)
        
        # Validation performance
        val_pred = model.predict(X_val)
        val_pred = np.clip(val_pred, 0, None)  # RUL cannot be negative
        val_metrics = evaluate_predictions(y_val, val_pred)
        
        # Test performance (against RUL_FD001.txt)
        test_pred = model.predict(X_test)
        test_pred = np.clip(test_pred, 0, None)  # RUL cannot be negative
        test_metrics = evaluate_predictions(y_test, test_pred)
        
        test_preds[name] = test_pred
        preds_dict_for_plot[name] = test_pred
        
        # Save model checkpoint
        slug = name.lower().replace(" ", "_")
        save_path = MODELS_DIR / f"{slug}.joblib"
        joblib.dump(model, save_path)
        print(f"Saved {name} model checkpoint to {save_path}")
        
        results.append({
            "Model": name,
            "Val_RMSE": val_metrics["RMSE"],
            "Val_MAE": val_metrics["MAE"],
            "Val_R2": val_metrics["R2"],
            "Test_RMSE": test_metrics["RMSE"],
            "Test_MAE": test_metrics["MAE"],
            "Test_R2": test_metrics["R2"],
            "Test_NASA_Score": test_metrics["NASA_Score"],
            "Test_Early_Preds": test_metrics["Early_Predictions"],
            "Test_Late_Preds": test_metrics["Late_Predictions"]
        })
        
        print(f"Validation: RMSE={val_metrics['RMSE']:.2f}, MAE={val_metrics['MAE']:.2f}, R²={val_metrics['R2']:.4f}")
        print(f"Test Set:   RMSE={test_metrics['RMSE']:.2f}, MAE={test_metrics['MAE']:.2f}, R²={test_metrics['R2']:.4f}, NASA Score={test_metrics['NASA_Score']:.1f}")

    # Summary DataFrame
    results_df = pd.DataFrame(results)
    results_csv_path = REPORTS_DIR / "classical_models_comparison.csv"
    results_df.to_csv(results_csv_path, index=False)
    print("\n==========================================")
    print(" Classical Models Performance Summary:")
    print("==========================================")
    print(results_df.to_string(index=False))

    # Save test predictions table
    test_preds_df = pd.DataFrame(test_preds)
    test_preds_df.to_csv(REPORTS_DIR / "classical_test_predictions.csv", index=False)

    # Plot feature importances & test predictions comparison
    plot_feature_importances(
        models["Random Forest"],
        models["XGBoost"],
        feature_names,
        FIGURES_DIR / "05_feature_importance.png"
    )
    
    plot_predictions_vs_actual(
        y_test,
        preds_dict_for_plot,
        FIGURES_DIR / "06_test_predictions_comparison.png"
    )

    return results_df


if __name__ == "__main__":
    train_classical_models("FD001")
