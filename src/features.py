"""
Feature engineering module for C-MAPSS Remaining Useful Life (RUL) prediction.
Implements:
1. Temporal rolling statistics (rolling mean, rolling std, instantaneous values)
   for classical tabular models (Linear Regression, Random Forest, XGBoost).
2. 3D sliding sequence window tensor generation (N, window_size, n_features)
   for Deep Learning models (LSTM, GRU, 1D-CNN).
"""

from pathlib import Path
import sys
from typing import Tuple, List
import pandas as pd
import numpy as np

# Ensure parent directory is on sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.config import (
    PROCESSED_DATA_DIR, FD001_ACTIVE_SENSORS, WINDOW_SIZE
)


# ==============================================================================
# 1. TABULAR ROLLING FEATURES (FOR CLASSICAL ML: LR, RF, XGBOOST)
# ==============================================================================

def create_tabular_features(
    df: pd.DataFrame,
    feature_cols: List[str] = FD001_ACTIVE_SENSORS,
    window_size: int = WINDOW_SIZE
) -> pd.DataFrame:
    """
    Compute rolling mean, rolling standard deviation, and instantaneous values
    per engine trajectory.

    Why rolling features?
    Individual sensor snapshots contain high-frequency measurement noise.
    - Rolling Mean: smooths noise and captures the underlying thermodynamic trajectory.
    - Rolling Std: captures variance expansion (as machinery wears, vibration and
      fluctuations increase).
    - Current Value: anchors the prediction to the current operational cycle.
    """
    df = df.copy()
    feature_dfs = []

    # Group strictly by engine unit_number so rolling windows never bleed across engines
    for unit, group in df.groupby("unit_number"):
        group = group.sort_values("time_cycles").copy()
        
        # Calculate rolling statistics
        roll = group[feature_cols].rolling(window=window_size, min_periods=1)
        rolling_mean = roll.mean().add_suffix(f"_roll_mean_{window_size}")
        rolling_std = roll.std().fillna(0).add_suffix(f"_roll_std_{window_size}")
        
        # Combine original columns with engineered features
        engineered = pd.concat([group, rolling_mean, rolling_std], axis=1)
        feature_dfs.append(engineered)

    combined_df = pd.concat(feature_dfs, ignore_index=True)
    return combined_df


def prepare_classical_datasets(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    rul_true: np.ndarray,
    feature_cols: List[str] = FD001_ACTIVE_SENSORS,
    window_size: int = WINDOW_SIZE
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, np.ndarray, List[str]]:
    """
    Generate feature matrices and target vectors for classical ML models.
    For test data, extracts the final cycle per engine to predict remaining life at cutoff.
    """
    # 1. Engineer rolling features
    train_eng = create_tabular_features(train_df, feature_cols, window_size)
    val_eng = create_tabular_features(val_df, feature_cols, window_size)
    test_eng = create_tabular_features(test_df, feature_cols, window_size)

    # 2. Identify all model input columns (raw sensors + rolling stats)
    engineered_cols = (
        feature_cols +
        [f"{col}_roll_mean_{window_size}" for col in feature_cols] +
        [f"{col}_roll_std_{window_size}" for col in feature_cols]
    )

    # Filter out initial cycles (< window_size) in training to ensure full statistical maturity
    train_filtered = train_eng[train_eng["time_cycles"] >= window_size]
    val_filtered = val_eng[val_eng["time_cycles"] >= window_size]

    X_train = train_filtered[engineered_cols]
    y_train = train_filtered["RUL_capped"]

    X_val = val_filtered[engineered_cols]
    y_val = val_filtered["RUL_capped"]

    # For the test set: extract the final recorded cycle per test engine
    test_last_cycles = test_eng.groupby("unit_number").last().reset_index()
    X_test = test_last_cycles[engineered_cols]
    y_test = rul_true  # Ground truth RUL from RUL_FDxxx.txt

    return X_train, y_train, X_val, y_val, X_test, y_test, engineered_cols


# ==============================================================================
# 2. 3D SLIDING SEQUENCE WINDOWS (FOR DEEP LEARNING: PYTORCH LSTM)
# ==============================================================================

def create_sequence_windows(
    df: pd.DataFrame,
    feature_cols: List[str] = FD001_ACTIVE_SENSORS,
    window_size: int = WINDOW_SIZE,
    target_col: str = "RUL_capped"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate 3D sliding sequence windows of shape (N, window_size, n_features).
    Used for training and validating Recurrent Neural Networks (LSTM/GRU).
    """
    sequences = []
    targets = []

    for unit, group in df.groupby("unit_number"):
        group = group.sort_values("time_cycles")
        feature_vals = group[feature_cols].values
        target_vals = group[target_col].values

        n_cycles = len(group)
        if n_cycles < window_size:
            continue

        for i in range(window_size, n_cycles + 1):
            seq = feature_vals[i - window_size:i, :]
            label = target_vals[i - 1]
            sequences.append(seq)
            targets.append(label)

    return np.array(sequences, dtype=np.float32), np.array(targets, dtype=np.float32)


def create_test_last_sequence_windows(
    test_df: pd.DataFrame,
    rul_true: np.ndarray,
    feature_cols: List[str] = FD001_ACTIVE_SENSORS,
    window_size: int = WINDOW_SIZE
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract the final sequence window of length `window_size` for each test engine.
    Matches test evaluation against true remaining useful life vector.
    """
    sequences = []
    valid_units = []

    for unit, group in test_df.groupby("unit_number"):
        group = group.sort_values("time_cycles")
        feature_vals = group[feature_cols].values
        n_cycles = len(group)

        if n_cycles >= window_size:
            seq = feature_vals[-window_size:, :]
        else:
            # If an engine has fewer cycles than window_size, pad with earliest cycle
            pad_len = window_size - n_cycles
            pad = np.tile(feature_vals[0, :], (pad_len, 1))
            seq = np.vstack([pad, feature_vals])

        sequences.append(seq)
        valid_units.append(unit)

    return np.array(sequences, dtype=np.float32), np.array(rul_true, dtype=np.float32)


def run_feature_generation(subset: str = "FD001"):
    """Orchestrate feature engineering and save matrices to disk."""
    print(f"\n==========================================")
    print(f" Running Feature Engineering on {subset}")
    print(f"==========================================")

    train_path = PROCESSED_DATA_DIR / f"{subset}_train_scaled.parquet"
    val_path = PROCESSED_DATA_DIR / f"{subset}_val_scaled.parquet"
    test_path = PROCESSED_DATA_DIR / f"{subset}_test_scaled.parquet"
    rul_path = PROCESSED_DATA_DIR / f"{subset}_rul_true.npy"

    if not train_path.exists():
        raise FileNotFoundError(f"Run preprocess.py first! Missing {train_path}")

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    test_df = pd.read_parquet(test_path)
    rul_true = np.load(rul_path)

    # 1. Classical ML Features
    X_train, y_train, X_val, y_val, X_test, y_test, feat_cols = prepare_classical_datasets(
        train_df, val_df, test_df, rul_true,
        feature_cols=FD001_ACTIVE_SENSORS, window_size=WINDOW_SIZE
    )
    print(f"Classical Tabular Datasets Generated:")
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_val:   {X_val.shape}, y_val:   {y_val.shape}")
    print(f"  X_test:  {X_test.shape}, y_test:  {y_test.shape}")
    print(f"  Total Engineered Features: {len(feat_cols)} (14 raw + 14 rolling mean + 14 rolling std)")

    # 2. Deep Learning 3D Windows
    X_seq_train, y_seq_train = create_sequence_windows(train_df, FD001_ACTIVE_SENSORS, WINDOW_SIZE)
    X_seq_val, y_seq_val = create_sequence_windows(val_df, FD001_ACTIVE_SENSORS, WINDOW_SIZE)
    X_seq_test, y_seq_test = create_test_last_sequence_windows(test_df, rul_true, FD001_ACTIVE_SENSORS, WINDOW_SIZE)

    print(f"\nDeep Learning 3D Sequence Windows Generated:")
    print(f"  X_seq_train: {X_seq_train.shape}, y_seq_train: {y_seq_train.shape}")
    print(f"  X_seq_val:   {X_seq_val.shape}, y_seq_val:   {y_seq_val.shape}")
    print(f"  X_seq_test:  {X_seq_test.shape}, y_seq_test:  {y_seq_test.shape}")

    # Persist classical features
    X_train.to_parquet(PROCESSED_DATA_DIR / f"{subset}_X_train_tabular.parquet")
    y_train.to_frame("RUL").to_parquet(PROCESSED_DATA_DIR / f"{subset}_y_train_tabular.parquet")
    X_val.to_parquet(PROCESSED_DATA_DIR / f"{subset}_X_val_tabular.parquet")
    y_val.to_frame("RUL").to_parquet(PROCESSED_DATA_DIR / f"{subset}_y_val_tabular.parquet")
    X_test.to_parquet(PROCESSED_DATA_DIR / f"{subset}_X_test_tabular.parquet")
    np.save(PROCESSED_DATA_DIR / f"{subset}_y_test_tabular.npy", y_test)

    # Persist sequence arrays
    np.save(PROCESSED_DATA_DIR / f"{subset}_X_seq_train.npy", X_seq_train)
    np.save(PROCESSED_DATA_DIR / f"{subset}_y_seq_train.npy", y_seq_train)
    np.save(PROCESSED_DATA_DIR / f"{subset}_X_seq_val.npy", X_seq_val)
    np.save(PROCESSED_DATA_DIR / f"{subset}_y_seq_val.npy", y_seq_val)
    np.save(PROCESSED_DATA_DIR / f"{subset}_X_seq_test.npy", X_seq_test)
    np.save(PROCESSED_DATA_DIR / f"{subset}_y_seq_test.npy", y_seq_test)

    print(f"\nAll processed features saved successfully in {PROCESSED_DATA_DIR}")


if __name__ == "__main__":
    run_feature_generation("FD001")
