"""
Comprehensive evaluation and benchmarking module for C-MAPSS RUL prediction.
Consolidates all 5 models:
1. Linear Regression (Baseline)
2. Random Forest Regressor
3. XGBoost Regressor
4. Turbofan1DCNN (Temporal Conv)
5. TurbofanLSTM (Recurrent Sequence)

Generates master comparison tables, residual error distributions,
and multi-engine test prediction trajectory visualizations.
"""

from pathlib import Path
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure parent directory is on sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.config import REPORTS_DIR, FIGURES_DIR, PROCESSED_DATA_DIR
from src.evaluate import evaluate_predictions, compute_nasa_score


def load_all_predictions(subset: str = "FD001") -> Tuple[pd.DataFrame, np.ndarray]:
    """Merge classical and deep learning test predictions."""
    classical_path = REPORTS_DIR / "classical_test_predictions.csv"
    deep_path = REPORTS_DIR / "deep_learning_test_predictions.csv"
    rul_path = PROCESSED_DATA_DIR / f"{subset}_rul_true.npy"

    if not classical_path.exists() or not deep_path.exists():
        raise FileNotFoundError("Missing test prediction CSVs in reports/! Run train.py and train_deep.py first.")

    df_class = pd.read_csv(classical_path)
    df_deep = pd.read_csv(deep_path)
    y_test = np.load(rul_path)

    merged = pd.DataFrame({
        "unit_number": df_class["Unit"],
        "Ground_Truth_RUL": y_test,
        "Linear Regression": df_class["Linear Regression"],
        "Random Forest": df_class["Random Forest"],
        "XGBoost": df_class["XGBoost"],
        "1D-CNN": df_deep["1D-CNN (Temporal Conv)"],
        "LSTM": df_deep["LSTM (Recurrent)"]
    })

    return merged, y_test


def compute_master_metrics(df_preds: pd.DataFrame, y_test: np.ndarray) -> pd.DataFrame:
    """Compute complete benchmark comparison metrics for all 5 models."""
    models = ["Linear Regression", "Random Forest", "XGBoost", "1D-CNN", "LSTM"]
    records = []

    for name in models:
        preds = df_preds[name].values
        m = evaluate_predictions(y_test, preds)
        
        diff = preds - y_test
        max_late = float(np.max(np.maximum(0, diff)))
        max_early = float(np.max(np.maximum(0, -diff)))

        records.append({
            "Model": name,
            "Family": "Baseline" if name == "Linear Regression" else ("Tree Ensemble" if name in ["Random Forest", "XGBoost"] else "Deep Learning"),
            "RMSE": m["RMSE"],
            "MAE": m["MAE"],
            "R2": m["R2"],
            "NASA_Score": m["NASA_Score"],
            "Early_Preds (Safe)": m["Early_Predictions"],
            "Late_Preds (Risk)": m["Late_Predictions"],
            "Max_Late_Error (Cycles)": round(max_late, 1),
            "Max_Early_Error (Cycles)": round(max_early, 1)
        })

    master_df = pd.DataFrame(records)
    return master_df


def plot_master_comparison_bars(master_df: pd.DataFrame, output_path: Path):
    """Plot 3-panel bar chart comparing models on RMSE, MAE, and NASA Score."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    palette = ["#7f8c8d", "#2980b9", "#27ae60", "#8e44ad", "#e74c3c"]

    # 1. RMSE
    sns.barplot(data=master_df, x="Model", y="RMSE", palette=palette, ax=axes[0], edgecolor="black")
    axes[0].set_title("Test Root Mean Squared Error (RMSE)", fontsize=11, fontweight="semibold")
    axes[0].set_ylabel("RMSE (Lower is Better)")
    axes[0].tick_params(axis="x", rotation=30)
    for p in axes[0].patches:
        axes[0].annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha="center", va="center", xytext=(0, 5), textcoords="offset points", fontsize=9)

    # 2. MAE
    sns.barplot(data=master_df, x="Model", y="MAE", palette=palette, ax=axes[1], edgecolor="black")
    axes[1].set_title("Test Mean Absolute Error (MAE)", fontsize=11, fontweight="semibold")
    axes[1].set_ylabel("MAE (Lower is Better)")
    axes[1].tick_params(axis="x", rotation=30)
    for p in axes[1].patches:
        axes[1].annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha="center", va="center", xytext=(0, 5), textcoords="offset points", fontsize=9)

    # 3. NASA Score
    sns.barplot(data=master_df, x="Model", y="NASA_Score", palette=palette, ax=axes[2], edgecolor="black")
    axes[2].set_title("NASA Asymmetric Scoring Function", fontsize=11, fontweight="semibold")
    axes[2].set_ylabel("Penalty Score (Lower is Better)")
    axes[2].tick_params(axis="x", rotation=30)
    for p in axes[2].patches:
        axes[2].annotate(f"{p.get_height():.1f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha="center", va="center", xytext=(0, 5), textcoords="offset points", fontsize=9)

    plt.suptitle("Master Benchmark: Model Performance Across Standard and Safety-Critical Metrics", fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved master comparison bar chart to {output_path}")


def plot_multi_model_test_curves(df_preds: pd.DataFrame, y_test: np.ndarray, output_path: Path):
    """Plot sorted ground truth vs model predictions across all 100 test engines."""
    sort_idx = np.argsort(y_test)
    y_sorted = y_test[sort_idx]

    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.plot(np.arange(100), y_sorted, label="Ground Truth (RUL_FD001.txt)", color="black", linewidth=2.5, linestyle="--")

    colors = {
        "Linear Regression": ("#7f8c8d", 1.2, ":"),
        "Random Forest": ("#2980b9", 1.4, "-."),
        "XGBoost": ("#27ae60", 1.6, "-"),
        "1D-CNN": ("#8e44ad", 1.3, "--"),
        "LSTM": ("#e74c3c", 2.2, "-")
    }

    for name, (color, lw, ls) in colors.items():
        preds_sorted = df_preds[name].values[sort_idx]
        ax.plot(np.arange(100), preds_sorted, label=name, color=color, linewidth=lw, linestyle=ls, alpha=0.85)

    ax.set_title("Test Engine Predictions vs Ground Truth (Ranked by Actual RUL)", fontsize=12, fontweight="semibold")
    ax.set_xlabel("100 Test Engines (Ordered from Lowest to Highest Remaining Life)")
    ax.set_ylabel("Remaining Useful Life (Cycles)")
    ax.legend(frameon=True, loc="upper left")
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved multi-model test curves to {output_path}")


def plot_residual_error_distribution(df_preds: pd.DataFrame, y_test: np.ndarray, output_path: Path):
    """
    Plot residual error distributions (Prediction - Ground Truth)
    highlighting the asymmetric safety zones.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
    axes = axes.flatten()

    models = ["Linear Regression", "Random Forest", "XGBoost", "1D-CNN", "LSTM"]
    colors = ["#7f8c8d", "#2980b9", "#27ae60", "#8e44ad", "#e74c3c"]

    for idx, (name, color) in enumerate(zip(models, colors)):
        ax = axes[idx]
        error = df_preds[name].values - y_test

        # Shaded background zones: Green for safe early prediction, Red for dangerous late prediction
        ax.axvspan(-60, 0, color="#27ae60", alpha=0.1, label="Safe (Early Maintenance)")
        ax.axvspan(0, 60, color="#c0392b", alpha=0.1, label="Catastrophic Risk (Late)")

        sns.histplot(error, bins=15, kde=True, color=color, ax=ax, edgecolor="black")
        ax.axvline(0, color="black", linestyle="--", linewidth=1.5)
        ax.axvline(np.mean(error), color="blue", linestyle=":", linewidth=1.2, label=f"Mean: {np.mean(error):+.1f}")

        ax.set_title(f"{name} Residuals (d = ŷ - y)", fontsize=10, fontweight="semibold")
        ax.set_xlabel("Prediction Error (Cycles)")
        ax.set_ylabel("Count of Engines")
        if idx == 0:
            ax.legend(fontsize=8, loc="upper right")

    # Hide unused 6th subplot
    axes[5].axis("off")

    plt.suptitle("Residual Error Distributions: Evaluating Early vs Late Asymmetric Risk Bias", fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved residual distribution plot to {output_path}")


def plot_sample_engine_predictions(df_preds: pd.DataFrame, output_path: Path, sample_units=(24, 31, 47, 68, 99)):
    """
    Plot actual vs predicted RUL bar comparison for representative test engines.
    """
    sample_df = df_preds[df_preds["unit_number"].isin(sample_units)].copy()

    melted = sample_df.melt(
        id_vars=["unit_number", "Ground_Truth_RUL"],
        value_vars=["Linear Regression", "Random Forest", "XGBoost", "LSTM"],
        var_name="Model", value_name="Predicted_RUL"
    )

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(sample_units))
    width = 0.16

    # Plot Ground Truth bars
    ax.bar(x - 2 * width, sample_df["Ground_Truth_RUL"], width, label="Ground Truth", color="black", edgecolor="black")

    # Plot Model bars
    model_colors = {"Linear Regression": "#7f8c8d", "Random Forest": "#2980b9", "XGBoost": "#27ae60", "LSTM": "#e74c3c"}
    for idx, (m_name, color) in enumerate(model_colors.items()):
        vals = sample_df[m_name].values
        ax.bar(x - width + idx * width, vals, width, label=m_name, color=color, edgecolor="black")

    ax.set_xticks(x)
    ax.set_xticklabels([f"Engine {u}" for u in sample_units], fontsize=10)
    ax.set_title("Predicted vs Actual RUL for Sample Test Engines (24, 31, 47, 68, 99)", fontsize=11, fontweight="semibold")
    ax.set_ylabel("Remaining Useful Life (Cycles)")
    ax.legend(frameon=True, loc="upper left")
    ax.grid(True, linestyle=":", alpha=0.5, axis="y")
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved sample engine predictions plot to {output_path}")


def run_master_comparison(subset: str = "FD001"):
    """Execute master comparative benchmarking and generate all diagnostic artifacts."""
    print(f"\n==========================================")
    print(f" Executing Master Evaluation & Comparison")
    print(f"==========================================")

    df_preds, y_test = load_all_predictions(subset)
    master_df = compute_master_metrics(df_preds, y_test)

    # Save CSV tables
    master_csv = REPORTS_DIR / "master_benchmark_summary.csv"
    master_df.to_csv(master_csv, index=False)
    
    preds_csv = REPORTS_DIR / "master_test_predictions.csv"
    df_preds.to_csv(preds_csv, index=False)

    print("\nMaster Benchmark Table:")
    print(master_df.to_string(index=False))

    # Generate diagnostic plots
    plot_master_comparison_bars(master_df, FIGURES_DIR / "08_master_model_comparison.png")
    plot_multi_model_test_curves(df_preds, y_test, FIGURES_DIR / "09_multi_model_test_curves.png")
    plot_residual_error_distribution(df_preds, y_test, FIGURES_DIR / "10_residual_error_distribution.png")
    plot_sample_engine_predictions(df_preds, FIGURES_DIR / "11_sample_engine_rul_trajectories.png")

    print("\nStage 6 Evaluation complete! All tables and diagnostic plots saved.")
    return master_df


if __name__ == "__main__":
    run_master_comparison("FD001")
