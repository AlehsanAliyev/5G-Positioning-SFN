import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import os
import joblib

from feature_engineering2 import merge_signal_data, extract_features_and_labels
from models.structured_mlp import StructuredMLP

CHECKPOINT_DIR = "outputs/checkpoints_v2"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def train_model(X, y_scaled, input_dim, save_path, epochs=100, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StructuredMLP(input_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32),
                            torch.tensor(y_scaled, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            pred_coords, _ = model(batch_x)
            loss = criterion(pred_coords, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"🧠 Epoch {epoch} | Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"✅ Model saved to: {save_path}")


def main():
    print("📦 Loading real GPS-labeled signal data...")
    df = merge_signal_data()
    X, y, _ = extract_features_and_labels(df)

    print("⚖️ Normalizing GPS labels...")
    y_scaler = MinMaxScaler()
    y_scaled = y_scaler.fit_transform(y)

    joblib.dump(y_scaler, os.path.join(CHECKPOINT_DIR, "y_scaler_v2.pkl"))

    input_dim = X.shape[1]
    print(f"✅ Training samples: {len(X)} | Feature dimension: {input_dim}")

    save_path = os.path.join(CHECKPOINT_DIR, "mlp_real_gps_scaled_v2.pth")

    print("🚀 Starting model training...")
    train_model(X, y_scaled, input_dim, save_path)


if __name__ == "__main__":
    main()