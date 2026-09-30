"""
Deep Learning training pipeline using PyTorch:
1. TurbofanLSTM (Recurrent Neural Network with LSTM cells)
2. Turbofan1DCNN (1D Convolutional Neural Network)

Trains on sequence window tensors (N, 30, 14), monitors validation loss
on unseen validation engines, saves best checkpoints, and evaluates against
ground truth test engines.
"""

from pathlib import Path
import sys
import copy
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Ensure parent directory is on sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.config import (
    PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR, REPORTS_DIR,
    RANDOM_SEED, BATCH_SIZE, EPOCHS, LEARNING_RATE
)
from src.models import TurbofanLSTM, Turbofan1DCNN
from src.evaluate import evaluate_predictions


def get_device() -> torch.device:
    """Select MPS (Apple Silicon GPU), CUDA, or fallback to CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int = RANDOM_SEED):
    """Ensure full reproducibility across PyTorch and NumPy."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_sequence_data(subset: str = "FD001"):
    """Load pre-windowed 3D tensors from disk."""
    X_train = np.load(PROCESSED_DATA_DIR / f"{subset}_X_seq_train.npy")
    y_train = np.load(PROCESSED_DATA_DIR / f"{subset}_y_seq_train.npy")
    
    X_val = np.load(PROCESSED_DATA_DIR / f"{subset}_X_seq_val.npy")
    y_val = np.load(PROCESSED_DATA_DIR / f"{subset}_y_seq_val.npy")
    
    X_test = np.load(PROCESSED_DATA_DIR / f"{subset}_X_seq_test.npy")
    y_test = np.load(PROCESSED_DATA_DIR / f"{subset}_y_seq_test.npy")
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def train_single_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = EPOCHS,
    lr: float = LEARNING_RATE,
    patience: int = 10
) -> Tuple[nn.Module, List[float], List[float]]:
    """
    Train a PyTorch sequence model with SmoothL1 loss, Adam optimizer,
    learning rate decay, and validation early stopping.
    """
    model = model.to(device)
    criterion = nn.SmoothL1Loss()  # Huber loss: robust against noisy early sensor outliers
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)

    train_losses = []
    val_losses = []

    best_val_loss = float("inf")
    best_weights = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        # Training loop
        model.train()
        batch_train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Stabilize gradients
            optimizer.step()

            batch_train_losses.append(loss.item())

        avg_train_loss = float(np.mean(batch_train_losses))
        train_losses.append(avg_train_loss)

        # Validation loop
        model.eval()
        batch_val_losses = []
        with torch.no_grad():
            for X_val_b, y_val_b in val_loader:
                X_val_b = X_val_b.to(device)
                y_val_b = y_val_b.to(device)
                val_pred = model(X_val_b)
                val_loss = criterion(val_pred, y_val_b)
                batch_val_losses.append(val_loss.item())

        avg_val_loss = float(np.mean(batch_val_losses))
        val_losses.append(avg_val_loss)

        scheduler.step(avg_val_loss)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:02d}/{epochs:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Check early stopping improvement
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_weights = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  Early stopping triggered at epoch {epoch}. Restoring best model (Val Loss: {best_val_loss:.4f})")
                break

    model.load_state_dict(best_weights)
    return model, train_losses, val_losses


def predict_model(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    """Run inference over a DataLoader and return clipped non-negative predictions."""
    model.eval()
    all_preds = []
    with torch.no_grad():
        for X_batch, _ in loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch).cpu().numpy().flatten()
            all_preds.extend(preds)
    preds_arr = np.array(all_preds)
    return np.clip(preds_arr, 0, None)  # RUL >= 0


def plot_learning_curves(history: Dict[str, Dict[str, List[float]]], output_path: Path):
    """Plot train vs validation loss across training epochs for all deep learning models."""
    fig, axes = plt.subplots(1, len(history), figsize=(6.5 * len(history), 4.5))
    if len(history) == 1:
        axes = [axes]

    for ax, (model_name, losses) in zip(axes, history.items()):
        epochs_range = range(1, len(losses["train"]) + 1)
        ax.plot(epochs_range, losses["train"], label="Train Loss (Smooth L1)", color="#2980b9", linewidth=1.8)
        ax.plot(epochs_range, losses["val"], label="Val Loss (Smooth L1)", color="#e67e22", linewidth=1.8, linestyle="--")
        ax.set_title(f"{model_name} Learning Curve", fontsize=11, fontweight="semibold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Smooth L1 Loss")
        ax.legend(frameon=True)
        ax.grid(True, linestyle=":", alpha=0.6)

    plt.suptitle("Deep Learning Training & Validation Loss Across Epochs", fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved learning curves plot to {output_path}")


def train_deep_learning_models(subset: str = "FD001"):
    """Execute end-to-end training and evaluation for PyTorch deep learning models."""
    set_seed(RANDOM_SEED)
    device = get_device()
    print(f"\n==========================================")
    print(f" Training Deep Learning Models on {subset}")
    print(f" PyTorch Execution Device: {device}")
    print(f"==========================================")

    # 1. Load sequence data
    X_train, y_train, X_val, y_val, X_test, y_test = load_sequence_data(subset)
    print(f"Dataset Tensors: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")

    # 2. PyTorch DataLoaders
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

    # 3. Model definitions
    input_dim = X_train.shape[2]  # 14 active sensors
    models_to_train = {
        "LSTM (Recurrent)": TurbofanLSTM(input_dim=input_dim, hidden_dim=64, num_layers=2, dropout=0.2),
        "1D-CNN (Temporal Conv)": Turbofan1DCNN(input_dim=input_dim, dropout=0.2)
    }

    history = {}
    results = []
    test_preds_dict = {"Unit": np.arange(1, len(y_test) + 1), "Ground_Truth_RUL": y_test}

    for name, model in models_to_train.items():
        print(f"\n--- Training {name} ---")
        trained_model, train_loss, val_loss = train_single_model(
            model, train_loader, val_loader, device=device,
            epochs=EPOCHS, lr=LEARNING_RATE, patience=10
        )
        history[name] = {"train": train_loss, "val": val_loss}

        # Evaluate on validation set
        val_preds = predict_model(trained_model, val_loader, device)
        val_metrics = evaluate_predictions(y_val, val_preds)

        # Evaluate on test set (100 test engines)
        test_preds = predict_model(trained_model, test_loader, device)
        test_metrics = evaluate_predictions(y_test, test_preds)
        test_preds_dict[name] = test_preds

        # Save model checkpoint
        slug = name.split()[0].lower().replace("(", "").replace(")", "").replace("-", "")
        save_path = MODELS_DIR / f"{slug}_model.pt"
        torch.save(trained_model.state_dict(), save_path)
        print(f"Saved {name} PyTorch checkpoint to {save_path}")

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
    results_csv_path = REPORTS_DIR / "deep_models_comparison.csv"
    results_df.to_csv(results_csv_path, index=False)
    print("\n==========================================")
    print(" Deep Learning Models Performance Summary:")
    print("==========================================")
    print(results_df.to_string(index=False))

    # Save test predictions table
    test_preds_df = pd.DataFrame(test_preds_dict)
    test_preds_df.to_csv(REPORTS_DIR / "deep_learning_test_predictions.csv", index=False)

    # Plot learning curves
    plot_learning_curves(history, FIGURES_DIR / "07_deep_learning_curves.png")

    return results_df


if __name__ == "__main__":
    train_deep_learning_models("FD001")
