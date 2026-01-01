from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import joblib
import numpy as np
import random

from models.structured_mlp import StructuredMLP
from feature_engineering2 import build_dataset, extract_features_and_labels

BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = BASE_DIR / "outputs" / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
SEED = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_model(X, y_scaled, input_dim, save_path, epochs=100, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StructuredMLP(input_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y_scaled, dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            pred_coords, _ = model(batch_x)
            loss = criterion(pred_coords, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch} | Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"Model saved to: {save_path}")
    return model


def main():
    set_seed(SEED)
    print("Loading DL dataset with engineered features...")
    df = build_dataset()
    X, y, x_scaler = extract_features_and_labels(df, scale=True)

    print("Normalizing GPS labels...")
    y_scaler = MinMaxScaler()
    y_scaled = y_scaler.fit_transform(y)

    joblib.dump(x_scaler, CHECKPOINT_DIR / "x_scaler_dl_geo.pkl")
    joblib.dump(y_scaler, CHECKPOINT_DIR / "y_scaler_dl_geo.pkl")

    input_dim = X.shape[1]
    print(f"Training samples: {len(X)} | Feature dimension: {input_dim}")

    save_path = CHECKPOINT_DIR / "mlp_dl_geo.pth"

    print("Starting model training...")
    train_model(X, y_scaled, input_dim, save_path)


if __name__ == "__main__":
    main()
