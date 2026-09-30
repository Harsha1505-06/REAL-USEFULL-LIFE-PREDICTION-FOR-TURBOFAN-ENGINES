"""
Model architectures and factory constructors for RUL prediction.
Includes:
1. Linear Regression (Baseline benchmark)
2. Random Forest Regressor (Non-linear decision tree ensemble)
3. XGBoost Regressor (Gradient-boosted decision trees)
4. PyTorch LSTM Regressor (Recurrent sequence network for temporal degradation)
"""

from typing import Dict, Any
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import torch
import torch.nn as nn


# ==============================================================================
# CLASSICAL MODELS (SCIKIT-LEARN & XGBOOST)
# ==============================================================================

def get_linear_regression() -> LinearRegression:
    """
    Baseline model: Ordinary Least Squares (OLS) Linear Regression.
    
    Why use Linear Regression as a baseline?
    In academic research and viva defense, complex models must justify their computational
    overhead against a parsimonious linear benchmark. If deep learning cannot decisively
    beat linear regression, the complexity is unwarranted.
    """
    return LinearRegression()


def get_random_forest(
    n_estimators: int = 100,
    max_depth: int = 12,
    random_state: int = 42
) -> RandomForestRegressor:
    """
    Classical ensemble: Bagging of decision trees.
    
    Why Random Forest?
    - Handles non-linear feature relationships without requiring linear separability.
    - Robust against overfitting due to bootstrap aggregating (bagging) and random feature subsampling.
    - Provides built-in Gini/impurity feature importance rankings.
    """
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1
    )


def get_xgboost(
    n_estimators: int = 100,
    max_depth: int = 5,
    learning_rate: float = 0.05,
    random_state: int = 42
) -> xgb.XGBRegressor:
    """
    Classical gradient boosting: Sequential residual minimization.
    
    Why XGBoost?
    - Sequentially trains shallow trees to correct residual errors of preceding trees.
    - Uses second-order Taylor expansion (gradient and hessian) for precise tree splits.
    - Includes L1 and L2 regularization to control model variance on tabular features.
    """
    return xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=random_state,
        n_jobs=-1
    )


# ==============================================================================
# DEEP LEARNING MODEL (PYTORCH LSTM)
# ==============================================================================

class TurbofanLSTM(nn.Module):
    """
    PyTorch Recurrent Neural Network with Long Short-Term Memory (LSTM) cells.

    Architecture:
    Input: (batch_size, sequence_length=30, num_features=14)
    -> LSTM Layer 1 (hidden_dim=64, dropout=0.2)
    -> LSTM Layer 2 (hidden_dim=32, dropout=0.2)
    -> Fully Connected Layer (32 -> 16, ReLU)
    -> Output Linear Layer (16 -> 1 scalar RUL prediction)
    
    Why LSTM for RUL prediction?
    Unlike tabular models that see only a single summary statistic of a window,
    LSTM networks retain an internal memory cell (c_t) and hidden state (h_t)
    governed by input, forget, and output gates. This allows the network to learn
    the temporal trajectory and rate of change of degradation across time steps.
    """
    def __init__(
        self,
        input_dim: int = 14,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM sequence encoder
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Regressor head
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, input_dim)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Extract representation at the final time step of the window (seq_len - 1)
        last_step_out = lstm_out[:, -1, :]  # shape: (batch_size, hidden_dim)
        
        # Regression prediction
        out = self.fc(last_step_out)  # shape: (batch_size, 1)
        return out.squeeze(-1)


class Turbofan1DCNN(nn.Module):
    """
    1D Convolutional Neural Network for temporal degradation feature extraction.

    Architecture:
    Input: (batch_size, seq_len=30, num_features=14)
    -> Transpose to (batch_size, 14 channels, 30 sequence length)
    -> Conv1D (14 -> 32, kernel=3, padding=1) -> BatchNorm1d -> ReLU -> MaxPool1d(2)
    -> Conv1D (32 -> 64, kernel=3, padding=1) -> BatchNorm1d -> ReLU -> AdaptiveAvgPool1d(1)
    -> Flatten -> Linear(64 -> 32) -> ReLU -> Dropout(0.2) -> Linear(32 -> 1)

    Why 1D-CNN?
    1D-CNN scans sliding temporal filters across adjacent operational cycles, capturing
    local degradation patterns (e.g., sudden pressure drops or thermal acceleration)
    without sequential recurrence. It trains significantly faster than recurrent networks
    and acts as a strong architectural comparison.
    """
    def __init__(self, input_dim: int = 14, dropout: float = 0.2):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # 30 -> 15

            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)       # 15 -> 1
        )
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is (batch_size, seq_len, input_dim) -> transpose to (batch_size, input_dim, seq_len)
        x_trans = x.permute(0, 2, 1)
        feat = self.conv_block(x_trans)  # shape: (batch_size, 64, 1)
        feat = feat.squeeze(-1)          # shape: (batch_size, 64)
        out = self.fc(feat)              # shape: (batch_size, 1)
        return out.squeeze(-1)

