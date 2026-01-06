from pathlib import Path
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import joblib
import random

from models.structured_mlp import StructuredMLP
from feature_engineering3 import build_dataset, extract_features_and_labels, spatial_kmeans_split

BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = BASE_DIR / "outputs" / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

N_CLUSTERS = 20
VAL_FRAC = 0.1
TEST_FRAC = 0.1
MAX_FEATURES = 200
WEIGHT_DECAY = 1e-4
SEED = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_model(X, y_scaled, input_dim, save_path, epochs=150, lr=5e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StructuredMLP(input_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    criterion = nn.SmoothL1Loss()

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


def compute_bias(model, X, y_true, y_scaler):
    device = next(model.parameters()).device
    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
    with torch.no_grad():
        coords_pred, _ = model(X_tensor)
    coords_pred_np = coords_pred.cpu().numpy()
    coords_pred_inv = y_scaler.inverse_transform(coords_pred_np)
    bias = np.mean(coords_pred_inv - y_true, axis=0)
    return bias


def main():
    set_seed(SEED)
    print("Loading DL dataset with engineered features...")
    df = build_dataset()
    train_df, val_df, test_df = spatial_kmeans_split(
        df,
        n_clusters=N_CLUSTERS,
        val_frac=VAL_FRAC,
        test_frac=TEST_FRAC,
    )

    X_train, y_train, x_scaler, feature_cols = extract_features_and_labels(
        train_df,
        scale=True,
        max_features=MAX_FEATURES,
    )
    X_val, y_val, _x, _ = extract_features_and_labels(
        val_df,
        scale=True,
        scaler=x_scaler,
        feature_cols=feature_cols,
    )

    print("Normalizing GPS labels...")
    y_scaler = MinMaxScaler()
    y_train_scaled = y_scaler.fit_transform(y_train)

    joblib.dump(x_scaler, CHECKPOINT_DIR / "x_scaler_dl_geo.pkl")
    joblib.dump(y_scaler, CHECKPOINT_DIR / "y_scaler_dl_geo.pkl")
    (CHECKPOINT_DIR / "feature_cols.json").write_text(json.dumps(feature_cols), encoding="utf-8")

    input_dim = X_train.shape[1]
    print(
        f"Training samples: {len(X_train)} | Val samples: {len(X_val)} "
        f"| Test samples: {len(test_df)} | Feature dimension: {input_dim}"
    )

    save_path = CHECKPOINT_DIR / "mlp_dl_geo.pth"

    print("Starting model training...")
    model = train_model(X_train, y_train_scaled, input_dim, save_path)

    bias = compute_bias(model, X_train, y_train, y_scaler)
    np.save(CHECKPOINT_DIR / "bias.npy", bias)
    print(f"Saved bias correction: dx={bias[0]:.6f}, dy={bias[1]:.6f}")


if __name__ == "__main__":
    main()
