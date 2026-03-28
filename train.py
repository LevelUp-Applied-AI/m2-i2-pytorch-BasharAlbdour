"""
Integration 2 — PyTorch: Housing Price Prediction
Module 2 — Programming for AI & Data Science

Complete each section below. Remove the TODO: comments and pass statements
as you implement each section. Do not change the overall structure.

Before running this script, install PyTorch:
    pip install torch --index-url https://download.pytorch.org/whl/cpu
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ─── Model Definition ─────────────────────────────────────────────────────────


class HousingModel(nn.Module):
    """Neural network for predicting housing prices from property features.

    Architecture: Linear(5, 32) -> ReLU -> Linear(32, 1)
    """

    def __init__(self):
        """Define the model layers."""
        super().__init__()
        self.layer1 = nn.Linear(5, 32)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(32, 1)
        pass

    def forward(self, x):
        """Define the forward pass.

                Args:
                    x (torch.Tensor): Input tensor of shape (N, 5).
        (price prediction)
                Returns:
                    torch.Tensor: Predictions of shape (N, 1).
        """
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x


def MAE_R2(y_true, x_pred):

    mae = np.mean(np.abs(y_true - x_pred))

    ss_res = np.sum((y_true - x_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    return mae, r2


# ─── Main Training Script ─────────────────────────────────────────────────────


def main():
    """Load data, train HousingModel, and save predictions."""

    # ── 1. Load Data ──────────────────────────────────────────────────────────
    df = pd.read_csv("data/housing.csv")
    print("Dataframe shape", df.shape)

    # ── 2. Separate Features and Target ──────────────────────────────────────
    feature_cols = [
        "area_sqm",
        "bedrooms",
        "floor",
        "age_years",
        "distance_to_center_km",
    ]
    X = df[feature_cols]
    y = df[["price_jod"]]

    # ── 3. Standardize Features ───────────────────────────────────────────────
    X_mean = X.mean()
    X_std = X.std()
    X_scaled = (X - X_mean) / X_std
    y_mean = y.mean()
    y_std = y.std()
    y_scaled = (y - y_mean) / y_std

    split_data = int(0.8 * len(X_scaled))
    X_train, X_test = X_scaled[:split_data], X_scaled[split_data:]
    y_train, y_test = y_scaled[:split_data], y_scaled[split_data:]

    # ── 4. Convert to Tensors ─────────────────────────────────────────────────
    X_trained = torch.tensor(X_train.values, dtype=torch.float32)
    X_tested = torch.tensor(X_test.values, dtype=torch.float32)
    y_trained = torch.tensor(y_train.values, dtype=torch.float32)
    y_tested = torch.tensor(y_test.values, dtype=torch.float32)
    print(f"X trained shape: {X_trained.shape}")
    print(f"X tested shape: {X_tested.shape}")
    print(f"Y trained shape: {y_trained.shape}")
    print(f"Y tested shape: {y_tested.shape}")

    # ── 5. Instantiate Model, Loss, and Optimizer ─────────────────────────────
    model = HousingModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # ── 6. Training Loop ──────────────────────────────────────────────────────
    num_epochs = 100
    loss_history = []
    for epoch in range(num_epochs):
        predictions = model(X_trained)
        loss = criterion(predictions, y_trained)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())

        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}: Loss = {loss.item():.4f}")

    # ── 7. Save Predictions ───────────────────────────────────────────────────
    with torch.no_grad():
        train_preds = model(X_trained).numpy()
        test_preds = model(X_tested).numpy()

    y_train_np = y_trained.numpy()
    y_test_np = y_tested.numpy()

    y_std_val = y_std.values[0]
    y_mean_val = y_mean.values[0]

    test_preds_unscaled = test_preds.flatten() * y_std_val + y_mean_val
    y_test_unscaled = y_test_np.flatten() * y_std_val + y_mean_val
    y_trained_uncalled = y_train_np.flatten() * y_std_val + y_mean_val
    train_preds_unscalled = train_preds.flatten() * y_std_val + y_mean_val

    train_mae, train_r2 = MAE_R2(y_trained_uncalled, train_preds_unscalled)
    test_mae, test_r2 = MAE_R2(y_test_unscaled, test_preds_unscaled)

    print(f"Train MAE: {train_mae:.2f}, R2: {train_r2:.4f}")
    print(f"Test  MAE: {test_mae:.2f}, R2: {test_r2:.4f}")

    # 9. Save Predictions (TEST SET)
    results_df = pd.DataFrame(
        {"actual": y_test_np.flatten(), "predicted": test_preds.flatten()}
    )
    results_df.to_csv("predictions.csv", index=False)
    print("Saved predictions.csv")

    plt.figure()
    plt.scatter(y_test_unscaled, test_preds_unscaled)

    min_val = min(y_test_unscaled.min(), test_preds_unscaled.min())
    max_val = max(y_test_unscaled.max(), test_preds_unscaled.max())
    plt.plot([min_val, max_val], [min_val, max_val])

    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs Predicted")

    plt.savefig("plots/predictions_plot.png")
    plt.close()

    plt.figure()
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")

    plt.savefig("plots/loss_curve.png")
    plt.close()


if __name__ == "__main__":
    main()
