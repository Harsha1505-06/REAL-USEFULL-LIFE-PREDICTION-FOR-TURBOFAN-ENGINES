"""
Preprocessing module for C-MAPSS turbofan engine data.
Implements:
1. Ground-truth RUL target calculation for training engines.
2. Piecewise-linear RUL capping (threshold ~125 cycles).
3. Leakage-free MinMaxScaler (fitted ONLY on training engines).
4. Engine-level grouped train/validation split (zero cross-engine leakage).
"""

from pathlib import Path
import sys
from typing import Tuple, List
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

# Ensure parent directory is on sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.config import (
    PROCESSED_DATA_DIR, MODELS_DIR, FD001_ACTIVE_SENSORS,
    RUL_CAP, RANDOM_SEED, VAL_SIZE
)
from src.data_loader import load_dataset


def add_rul_targets(df: pd.DataFrame, rul_cap: int = RUL_CAP) -> pd.DataFrame:
    """
    Compute Remaining Useful Life (RUL) for each cycle of training engines.
    
    Why RUL capping?
    In early operating cycles, turbofan engines operate within nominal healthy tolerances.
    Sensors exhibit flat baselines and micro-damage is physically undetectable.
    Training a model to differentiate RUL=300 from RUL=250 forces it to fit random
    sensor noise. Capping RUL at a piecewise threshold (e.g., 125 cycles) aligns
    mathematical loss with the physical onset of measurable wear.
    """
    df = df.copy()
    max_cycle_per_unit = df.groupby("unit_number")["time_cycles"].transform("max")
    
    # Linear un-capped RUL
    df["RUL_true"] = max_cycle_per_unit - df["time_cycles"]
    
    # Piecewise-linear capped RUL
    df["RUL_capped"] = np.minimum(df["RUL_true"], rul_cap)
    
    return df


def split_train_val_by_engine(
    train_df: pd.DataFrame,
    val_size: float = VAL_SIZE,
    random_seed: int = RANDOM_SEED
) -> Tuple[pd.DataFrame, pd.DataFrame, List[int], List[int]]:
    """
    Partition training data into train and validation sets grouped STRICTLY by engine ID.
    
    Why split by engine ID, never by row?
    Time-series rows from the same engine share persistent physical manufacturing traits
    and wear profiles. Splitting randomly by row leaks future cycle information of an engine
    into the training fold, artificially inflating validation accuracy.
    """
    unique_units = train_df["unit_number"].unique()
    
    rng = np.random.RandomState(random_seed)
    shuffled_units = rng.permutation(unique_units)
    
    n_val = int(len(unique_units) * val_size)
    val_units = sorted(shuffled_units[:n_val].tolist())
    train_units = sorted(shuffled_units[n_val:].tolist())
    
    # Ensure zero overlap
    assert len(set(train_units).intersection(set(val_units))) == 0, "Engine ID leakage detected!"
    
    train_split = train_df[train_df["unit_number"].isin(train_units)].copy().reset_index(drop=True)
    val_split = train_df[train_df["unit_number"].isin(val_units)].copy().reset_index(drop=True)
    
    return train_split, val_split, train_units, val_units


def fit_and_apply_scaler(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str] = FD001_ACTIVE_SENSORS,
    scaler_save_path: Path = MODELS_DIR / "minmax_scaler.joblib"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, MinMaxScaler]:
    """
    Normalize feature columns using MinMaxScaler fitted STRICTLY on the training fold.
    
    Why MinMaxScaler over StandardScaler?
    1. Sensors operate within well-defined physical envelopes (e.g. pressure, temperature).
       MinMaxScaler maps these non-negative signals onto [0, 1] without distorting zero baselines.
    2. Neural network activation functions (tanh, sigmoid, relu) perform stably when inputs
       are bounded within [0, 1].
    3. Prevents high-magnitude sensors (e.g. core speed ~9000 rpm) from dominating
       low-magnitude sensors (e.g. bypass ratio ~8.4).
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    # FIT ONLY ON TRAIN
    scaler.fit(train_df[feature_cols])
    
    train_scaled = train_df.copy()
    val_scaled = val_df.copy()
    test_scaled = test_df.copy()
    
    # TRANSFORM ALL
    train_scaled[feature_cols] = scaler.transform(train_df[feature_cols])
    val_scaled[feature_cols] = scaler.transform(val_df[feature_cols])
    test_scaled[feature_cols] = scaler.transform(test_df[feature_cols])
    
    # Persist scaler for downstream inference & Streamlit app
    joblib.dump(scaler, scaler_save_path)
    print(f"Fitted MinMaxScaler on {len(feature_cols)} features and saved to {scaler_save_path}")
    
    return train_scaled, val_scaled, test_scaled, scaler


def run_preprocessing(subset: str = "FD001"):
    """Orchestrate end-to-end preprocessing pipeline for a dataset subset."""
    print(f"\n==========================================")
    print(f" Running Preprocessing Pipeline on {subset}")
    print(f"==========================================")
    
    train_raw, test_raw, rul_true = load_dataset(subset)
    
    # Step 1: Add RUL targets
    train_with_rul = add_rul_targets(train_raw, rul_cap=RUL_CAP)
    print(f"Calculated RUL targets for train set. Capped RUL max: {train_with_rul['RUL_capped'].max()}")
    
    # Step 2: Split by engine ID
    train_split, val_split, train_ids, val_ids = split_train_val_by_engine(
        train_with_rul, val_size=VAL_SIZE, random_seed=RANDOM_SEED
    )
    print(f"Grouped Engine Split: {len(train_ids)} train engines ({len(train_split)} cycles), "
          f"{len(val_ids)} val engines ({len(val_split)} cycles)")
    print(f"Validation Engine IDs: {val_ids}")
    
    # Step 3: Scale features (fitted ONLY on train)
    train_scaled, val_scaled, test_scaled, scaler = fit_and_apply_scaler(
        train_split, val_split, test_raw, feature_cols=FD001_ACTIVE_SENSORS
    )
    
    # Step 4: Persist processed DataFrames
    train_scaled.to_parquet(PROCESSED_DATA_DIR / f"{subset}_train_scaled.parquet", index=False)
    val_scaled.to_parquet(PROCESSED_DATA_DIR / f"{subset}_val_scaled.parquet", index=False)
    test_scaled.to_parquet(PROCESSED_DATA_DIR / f"{subset}_test_scaled.parquet", index=False)
    np.save(PROCESSED_DATA_DIR / f"{subset}_rul_true.npy", rul_true)
    
    print(f"Processed tabular datasets successfully saved to {PROCESSED_DATA_DIR}")
    return train_scaled, val_scaled, test_scaled, rul_true


if __name__ == "__main__":
    run_preprocessing("FD001")
