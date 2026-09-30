"""
Exploratory Data Analysis (EDA) module for NASA C-MAPSS dataset.
Generates statistical profiles, identifies invariant/zero-variance sensors,
examines sensor degradation trajectories across operational cycles,
and generates publication-quality diagnostic plots.
"""

from pathlib import Path
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure parent directory is on sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.config import (
    SENSOR_COLS, SETTING_COLS, SENSOR_INFO,
    FIGURES_DIR, FD001_DROP_SENSORS, FD001_ACTIVE_SENSORS
)
from src.data_loader import load_dataset, get_dataset_summary


def set_plot_style():
    """Apply clean, minimal, publication-style plotting aesthetics."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 13,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def plot_engine_lifespan_distribution(train_df: pd.DataFrame, output_path: Path):
    """Plot distribution of engine total operational cycles before failure."""
    engine_lifespans = train_df.groupby("unit_number")["time_cycles"].max()
    
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(engine_lifespans, bins=20, kde=True, color="#2b5c8f", ax=ax, edgecolor="black")
    
    ax.axvline(engine_lifespans.mean(), color="#d95f02", linestyle="--", linewidth=1.5,
               label=f"Mean: {engine_lifespans.mean():.1f} cycles")
    ax.axvline(engine_lifespans.median(), color="#7570b3", linestyle=":", linewidth=1.5,
               label=f"Median: {engine_lifespans.median():.1f} cycles")
    ax.axvline(engine_lifespans.min(), color="#1b9e77", linestyle="-.", linewidth=1.2,
               label=f"Min: {engine_lifespans.min()} | Max: {engine_lifespans.max()}")

    ax.set_title("Distribution of Engine Lifespans to Failure (FD001 Train)")
    ax.set_xlabel("Total Operational Cycles Until Failure (End-of-Life)")
    ax.set_ylabel("Engine Count")
    ax.legend(frameon=True)
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_sensor_variance(train_df: pd.DataFrame, output_path: Path):
    """Identify invariant (zero-variance) sensors vs active sensors."""
    stds = train_df[SENSOR_COLS].std()
    
    colors = ["#c0392b" if col in FD001_DROP_SENSORS else "#2980b9" for col in SENSOR_COLS]
    
    fig, ax = plt.subplots(figsize=(10, 4.5))
    bars = ax.bar(SENSOR_COLS, stds.values, color=colors, edgecolor="black", width=0.6)
    
    ax.set_yscale("log")
    ax.set_title("Standard Deviation of Sensor Measurements (FD001 Train, Log Scale)")
    ax.set_xlabel("Sensor Channel")
    ax.set_ylabel("Standard Deviation (Log Scale)")
    
    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2980b9", edgecolor="black", label="Active Informative Sensor"),
        Patch(facecolor="#c0392b", edgecolor="black", label=f"Flat / Zero-Variance Sensor (Dropped: {', '.join(FD001_DROP_SENSORS)})")
    ]
    ax.legend(handles=legend_elements, loc="upper right")
    plt.xticks(rotation=45)
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_sensor_trajectories(train_df: pd.DataFrame, output_path: Path, sample_units=(1, 2, 3, 4, 5)):
    """Plot degradation trajectories for key informative sensors across operational cycles."""
    # Top informative sensors exhibiting strong physical degradation trend
    key_sensors = [
        ("s_2", "LPC Outlet Temp (°R)"),
        ("s_3", "HPC Outlet Temp (°R)"),
        ("s_4", "LPT Outlet Temp (°R)"),
        ("s_7", "HPC Outlet Pressure (psia)"),
        ("s_11", "HPC Static Pressure (psia)"),
        ("s_12", "Fuel Flow Ratio (pps/psia)"),
        ("s_15", "Bypass Ratio"),
        ("s_20", "HPT Coolant Bleed (lbm/s)"),
        ("s_21", "LPT Coolant Bleed (lbm/s)"),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharex=False)
    axes = axes.flatten()

    colors = sns.color_palette("tab10", len(sample_units))

    for ax_idx, (sensor, label) in enumerate(key_sensors):
        ax = axes[ax_idx]
        for u_idx, unit in enumerate(sample_units):
            unit_data = train_df[train_df["unit_number"] == unit]
            ax.plot(
                unit_data["time_cycles"],
                unit_data[sensor],
                color=colors[u_idx],
                alpha=0.8,
                linewidth=1.2,
                label=f"Engine {unit}" if ax_idx == 0 else ""
            )
        ax.set_title(f"{sensor}: {label}", fontsize=10, fontweight="semibold")
        ax.set_xlabel("Time (Cycles)")
        ax.set_ylabel(label.split("(")[-1].replace(")", "").strip() if "(" in label else "Value")
        ax.grid(True, linestyle="--", alpha=0.5)

    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=len(sample_units),
               bbox_to_anchor=(0.5, 0.99), frameon=True)
    plt.suptitle("Sensor Degradation Trajectories Across Cycles for Engines 1–5 (FD001)", y=1.02, fontsize=13)
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_correlation_matrix(train_df: pd.DataFrame, output_path: Path):
    """Plot correlation heatmap between active sensors, operating cycles, and RUL."""
    # Compute ground-truth RUL for training set
    df_copy = train_df.copy()
    max_cycle = df_copy.groupby("unit_number")["time_cycles"].transform("max")
    df_copy["RUL"] = max_cycle - df_copy["time_cycles"]

    features_to_correlate = ["time_cycles"] + FD001_ACTIVE_SENSORS + ["RUL"]
    corr = df_copy[features_to_correlate].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        corr,
        cmap="coolwarm",
        center=0,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 7},
        cbar_kws={"label": "Pearson Correlation Coefficient"},
        ax=ax,
        linewidths=0.5
    )
    ax.set_title("Correlation Heatmap: Active Sensors, Operating Cycles, and RUL (FD001)", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved: {output_path}")


def run_eda(subset: str = "FD001"):
    """Execute complete EDA pipeline and save figures."""
    set_plot_style()
    print(f"\n==========================================")
    print(f" Running EDA Pipeline on {subset}")
    print(f"==========================================")
    
    train_df, test_df, rul_true = load_dataset(subset)
    summary = get_dataset_summary(train_df, test_df, rul_true)
    
    print("\nDataset Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    # Generate and save figures
    plot_engine_lifespan_distribution(train_df, FIGURES_DIR / "01_engine_lifespan_distribution.png")
    plot_sensor_variance(train_df, FIGURES_DIR / "02_sensor_variance_profile.png")
    plot_sensor_trajectories(train_df, FIGURES_DIR / "03_sensor_degradation_trajectories.png")
    plot_correlation_matrix(train_df, FIGURES_DIR / "04_sensor_correlation_matrix.png")

    print("\nEDA Completed successfully. All plots saved to:", FIGURES_DIR)


if __name__ == "__main__":
    run_eda("FD001")
