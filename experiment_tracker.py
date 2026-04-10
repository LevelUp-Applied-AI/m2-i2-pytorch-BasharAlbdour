import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import json
import time


class HousingModel(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.layer1 = nn.Linear(5, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
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


def main():

    np.random.seed(42)

    df = pd.read_csv("data/housing.csv")
    print("dataframe shape", df.shape)

    feature_cols = [
        "area_sqm",
        "bedrooms",
        "floor",
        "age_years",
        "distance_to_center_km",
    ]
    X = df[feature_cols]
    y = df[["price_jod"]]

    X_mean = X.mean()
    X_std = X.std()
    X_scaled = (X - X_mean) / X_std
    y_mean = y.mean()
    y_std = y.std()
    y_scaled = (y - y_mean) / y_std

    split_data = int(0.8 * len(X_scaled))
    X_train, X_test = X_scaled[:split_data], X_scaled[split_data:]
    y_train, y_test = y_scaled[:split_data], y_scaled[split_data:]

    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)
    print(f"X train tensor shape:{X_train_tensor.shape}")
    print(f"X test tensor shape:{X_test_tensor.shape}")
    print(f"y train tensor shape:{y_train_tensor.shape}")
    print(f"y test tensor shape:{y_test_tensor.shape}")

    learning_rates = [0.001, 0.01, 0.05]
    hidden_sizes = [16, 32, 64]
    epochs_list = [50, 100, 150]

    results = []

    for lr in learning_rates:
        for hidden in hidden_sizes:
            for epochs in epochs_list:
                start_time = time.time()
                model = HousingModel(hidden)
                criterion = nn.MSELoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                print(f"lr:{lr}, hidden:{hidden}, epochs:{epochs}")

                for epoch in range(epochs):
                    preds = model(X_train_tensor)
                    loss = criterion(preds, y_train_tensor)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    if epoch % 20 == 0:
                        print(f"Epoch {epoch:3d}: Loss = {loss.item():.4f}")
                with torch.inference_mode():
                    train_preds = model(X_train_tensor).numpy()
                    test_preds = model(X_test_tensor).numpy()

                y_train_np = y_train_tensor.numpy()
                y_test_np = y_test_tensor.numpy()

                y_mean_val = y_mean.values[0]
                y_std_val = y_std.values[0]

                train_preds_unscaled = train_preds.flatten() * y_std_val + y_mean_val
                y_train_unscaled = y_train_np.flatten() * y_std_val + y_mean_val

                test_preds_unscaled = test_preds.flatten() * y_std_val + y_mean_val
                y_test_unscaled = y_test_np.flatten() * y_std_val + y_mean_val

                train_mae, train_r2 = MAE_R2(y_train_unscaled, train_preds_unscaled)
                test_mae, test_r2 = MAE_R2(y_test_unscaled, test_preds_unscaled)

                elapsed_time = time.time() - start_time

                results.append(
                    {
                        "hidden_size": hidden,
                        "learning_rate": lr,
                        "epochs": epochs,
                        "train_mae": train_mae,
                        "test_mae": test_mae,
                        "test_r2": test_r2,
                        "time_sec": elapsed_time,
                    }
                )

    results_sorted = sorted(results, key=lambda x: x["test_mae"])

    print("\n Top 10 Models:\n")

    print(
        f"{'Rank':<5} | {'LR':<7} | {'Hidden':<6} | {'Epochs':<6} | {'Test MAE':<12} | {'Test R²':<9} | {'Time (s)':<8}"
    )
    print("-" * 80)

    for i, r in enumerate(results_sorted[:10], start=1):
        print(
            f"{i:<5} | "
            f"{r['learning_rate']:<7} | "
            f"{r['hidden_size']:<6} | "
            f"{r['epochs']:<6} | "
            f"{r['test_mae']:<12.2f} | "
            f"{r['test_r2']:<9.4f} | "
            f"{r['time_sec']:<8.2f}"
        )

    with open("experiments.json", "w") as f:
        json.dump(results_sorted, f, indent=4)

    lrs = [r["learning_rate"] for r in results]
    maes = [r["test_mae"] for r in results]

    plt.figure()
    plt.scatter(lrs, maes)
    plt.xlabel("Learning Rate")
    plt.ylabel("Test MAE")
    plt.title("Learning Rate vs MAE")
    plt.savefig("experiment_summary.png")
    plt.close()

    print("\nSaved experiments.json and experiment_summary.png")


if __name__ == "__main__":
    main()
