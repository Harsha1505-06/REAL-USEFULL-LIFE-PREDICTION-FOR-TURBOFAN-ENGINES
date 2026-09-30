"""
Data loader module for NASA C-MAPSS Turbofan Engine datasets.
Handles space-separated text file ingestion, header assignment, schema validation,
and ground-truth RUL vector loading.
"""

from pathlib import Path
from typing import Tuple, Dict, Any
import pandas as pd
import numpy as np

import sys
# Allow importing config from parent directory
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.config import ALL_COLUMNS, DATASETS, RAW_DATA_DIR


def load_dataset(subset: str = "FD001") -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """
    Load train, test, and ground-truth RUL data for a specific C-MAPSS subset.

    Parameters
    ----------
    subset : str
        Dataset identifier ('FD001', 'FD002', 'FD003', or 'FD004').

    Returns
    -------
    train_df : pd.DataFrame
        Training telemetry dataframe with assigned 26 column headers.
    test_df : pd.DataFrame
        Test telemetry dataframe with assigned 26 column headers.
    rul_true : np.ndarray
        1D array containing ground truth remaining useful life for test units.
    """
    if subset not in DATASETS:
        raise ValueError(f"Unknown subset '{subset}'. Valid choices: {list(DATASETS.keys())}")

    meta = DATASETS[subset]
    train_path = meta["train"]
    test_path = meta["test"]
    rul_path = meta["rul"]

    if not train_path.exists():
        raise FileNotFoundError(f"Training file not found: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test file not found: {test_path}")
    if not rul_path.exists():
        raise FileNotFoundError(f"RUL ground truth file not found: {rul_path}")

    # C-MAPSS files are space-separated with no headers and trailing spaces
    train_df = pd.read_csv(train_path, sep=r"\s+", header=None, names=ALL_COLUMNS)
    test_df = pd.read_csv(test_path, sep=r"\s+", header=None, names=ALL_COLUMNS)
    
    # Ground truth RUL file has 1 integer per row corresponding to unit 1..N in order
    rul_df = pd.read_csv(rul_path, sep=r"\s+", header=None)
    rul_true = rul_df.values.flatten()

    # Validate shapes and null counts
    assert train_df.shape[1] == 26, f"Expected 26 columns in {subset} train, got {train_df.shape[1]}"
    assert test_df.shape[1] == 26, f"Expected 26 columns in {subset} test, got {test_df.shape[1]}"
    assert len(rul_true) == meta["test_units"], (
        f"Expected {meta['test_units']} ground truth RUL values in {subset}, got {len(rul_true)}"
    )
    assert train_df.isna().sum().sum() == 0, f"Missing values detected in {subset} train data"
    assert test_df.isna().sum().sum() == 0, f"Missing values detected in {subset} test data"

    return train_df, test_df, rul_true


def get_dataset_summary(train_df: pd.DataFrame, test_df: pd.DataFrame, rul_true: np.ndarray) -> Dict[str, Any]:
    """
    Compute engine count, cycle statistics, and trajectory lengths.
    """
    train_units = train_df["unit_number"].nunique()
    test_units = test_df["unit_number"].nunique()
    
    train_engine_cycles = train_df.groupby("unit_number")["time_cycles"].max()
    test_engine_cycles = test_df.groupby("unit_number")["time_cycles"].max()

    return {
        "train_units": train_units,
        "test_units": test_units,
        "train_total_cycles": len(train_df),
        "test_total_cycles": len(test_df),
        "train_cycle_min": int(train_engine_cycles.min()),
        "train_cycle_max": int(train_engine_cycles.max()),
        "train_cycle_mean": float(train_engine_cycles.mean()),
        "test_cycle_min": int(test_engine_cycles.min()),
        "test_cycle_max": int(test_engine_cycles.max()),
        "test_cycle_mean": float(test_engine_cycles.mean()),
        "rul_min": int(rul_true.min()),
        "rul_max": int(rul_true.max()),
        "rul_mean": float(rul_true.mean()),
    }


if __name__ == "__main__":
    train_df, test_df, rul_true = load_dataset("FD001")
    summary = get_dataset_summary(train_df, test_df, rul_true)
    print("FD001 Dataset Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
