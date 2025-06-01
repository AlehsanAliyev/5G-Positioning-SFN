# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset
# import os
# import numpy as np
# from tqdm import tqdm
# from src.models.structured_mlp import StructuredMLP
# from src.feature_engineering import merge_signal_data, extract_features
#
# CHECKPOINT_DIR = "outputs/checkpoints/"
# os.makedirs(CHECKPOINT_DIR, exist_ok=True)
#
# def simulate_labels(n_samples):
#     # Dummy (x, y) in campus bounds for supervised training
#     lat_range = (41.104, 41.107)
#     lon_range = (29.025, 29.030)
#     x = np.random.uniform(lon_range[0], lon_range[1], n_samples)
#     y = np.random.uniform(lat_range[0], lat_range[1], n_samples)
#     return np.stack([x, y], axis=1)
#
# def train_single_model(X, y, input_dim, save_path, epochs=50, lr=1e-3):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = StructuredMLP(input_dim=input_dim).to(device)
#     optimizer = optim.Adam(model.parameters(), lr=lr)
#     criterion = nn.MSELoss()
#
#     dataset = TensorDataset(torch.tensor(X, dtype=torch.float32),
#                             torch.tensor(y, dtype=torch.float32))
#     loader = DataLoader(dataset, batch_size=32, shuffle=True)
#
#     model.train()
#     for epoch in range(epochs):
#         total_loss = 0
#         for batch_x, batch_y in loader:
#             batch_x, batch_y = batch_x.to(device), batch_y.to(device)
#             pred_coords, _ = model(batch_x)
#             loss = criterion(pred_coords, batch_y)
#
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
#
#         if epoch % 10 == 0:
#             print(f"[Epoch {epoch}] Loss: {total_loss:.4f}")
#
#     torch.save(model.state_dict(), save_path)
#     print(f"✅ Model saved to {save_path}")
#
# def main():
#     print("🚀 Loading data...")
#     merged = merge_signal_data(stat_filter="Mean")
#     X_df = extract_features(merged)
#
#     # Simulate labels
#     y = simulate_labels(len(X_df))
#     X = X_df.fillna(X_df.median()).values
#     input_dim = X.shape[1]
#
#     print("🧠 Training base station model...")
#     save_path = os.path.join(CHECKPOINT_DIR, "structured_mlp.pth")
#     train_single_model(X, y, input_dim, save_path)
#
# if __name__ == "__main__":
#     main()



# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset
# import os
# import numpy as np
# import pandas as pd
# from tqdm import tqdm
#
# from models.structured_mlp import StructuredMLP
# from feature_engineering import merge_signal_data, extract_features
#
# CHECKPOINT_DIR = "outputs/checkpoints/"
# os.makedirs(CHECKPOINT_DIR, exist_ok=True)
#
# def simulate_labels(n_samples):
#     lat_range = (41.104, 41.107)
#     lon_range = (29.025, 29.030)
#     x = np.random.uniform(lon_range[0], lon_range[1], n_samples)
#     y = np.random.uniform(lat_range[0], lat_range[1], n_samples)
#     return np.stack([x, y], axis=1)
#
# def train_model(X, y, input_dim, save_path, epochs=30, lr=1e-3):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = StructuredMLP(input_dim).to(device)
#     optimizer = optim.Adam(model.parameters(), lr=lr)
#     criterion = nn.MSELoss()
#
#     dataset = TensorDataset(torch.tensor(X, dtype=torch.float32),
#                             torch.tensor(y, dtype=torch.float32))
#     loader = DataLoader(dataset, batch_size=16, shuffle=True)
#
#     model.train()
#     for epoch in range(epochs):
#         total_loss = 0
#         for batch_x, batch_y in loader:
#             batch_x, batch_y = batch_x.to(device), batch_y.to(device)
#             pred_coords, _ = model(batch_x)
#             loss = criterion(pred_coords, batch_y)
#
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
#
#         if epoch % 10 == 0:
#             print(f"  Epoch {epoch} | Loss: {total_loss:.4f}")
#
#     torch.save(model.state_dict(), save_path)
#     print(f"✅ Saved: {save_path}")
#
# def main():
#     print("📥 Loading signal data...")
#     merged = merge_signal_data(stat_filter="Mean")
#     features_df = extract_features(merged)
#
#     # Identify all serving PCI columns (e.g. "DL_NR_UE_PCI_0", "UL_*")
#     pci_cols = [col for col in features_df.columns if "PCI_0" in col]
#     if not pci_cols:
#         raise ValueError("❌ No PCI columns found in features")
#
#     print(f"📡 Found {len(pci_cols)} PCI columns: {pci_cols}")
#
#     for pci_col in pci_cols:
#         print(f"\n🧪 Training MLP for base station: {pci_col}")
#
#         # Select subset where this PCI appears (non-null)
#         pci_df = features_df[features_df[pci_col].notnull()]
#         if pci_df.empty:
#             print("  ⚠️ No data for this PCI. Skipping.")
#             continue
#
#         X = pci_df.fillna(pci_df.median()).values
#         y = simulate_labels(len(X))
#
#         save_path = os.path.join(CHECKPOINT_DIR, f"mlp_{pci_col}.pth")
#         train_model(X, y, input_dim=X.shape[1], save_path=save_path)
#
# if __name__ == "__main__":
#     main()

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset
# from sklearn.preprocessing import StandardScaler
# import os
# import numpy as np
# import pandas as pd
# from tqdm import tqdm
#
# from models.structured_mlp import StructuredMLP
# from feature_engineering import merge_signal_data, extract_features
#
# CHECKPOINT_DIR = "outputs/checkpoints/"
# os.makedirs(CHECKPOINT_DIR, exist_ok=True)
#
# def train_model(X, y, input_dim, save_path, epochs=50, lr=1e-3):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = StructuredMLP(input_dim).to(device)
#     optimizer = optim.Adam(model.parameters(), lr=lr)
#     criterion = nn.MSELoss()
#
#     dataset = TensorDataset(torch.tensor(X, dtype=torch.float32),
#                             torch.tensor(y, dtype=torch.float32))
#     loader = DataLoader(dataset, batch_size=32, shuffle=True)
#
#     model.train()
#     for epoch in range(epochs):
#         total_loss = 0
#         for batch_x, batch_y in loader:
#             batch_x, batch_y = batch_x.to(device), batch_y.to(device)
#             pred_coords, _ = model(batch_x)
#             loss = criterion(pred_coords, batch_y)
#
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
#
#         if epoch % 10 == 0:
#             print(f"  Epoch {epoch} | Loss: {total_loss:.4f}")
#
#     torch.save(model.state_dict(), save_path)
#     print(f"✅ Saved: {save_path}")
#
# def main():
#     print("📥 Loading signal data...")
#     merged = merge_signal_data(stat_filter="Mean")
#
#     if not all(col in merged.columns for col in ["lat", "lon"]):
#         raise ValueError("❌ GPS labels (lat/lon) not found in data.")
#
#     # Extract features and real labels
#     X_df = extract_features(merged)
#     X = X_df.fillna(X_df.median()).values
#     y = merged[["lon", "lat"]].values  # Real ground truth
#
#     # Normalize features
#     scaler = StandardScaler()
#     X = scaler.fit_transform(X)
#
#     input_dim = X.shape[1]
#     save_path = os.path.join(CHECKPOINT_DIR, "mlp_real_data.pth")
#
#     print("🧠 Training model using real labels...")
#     train_model(X, y, input_dim, save_path)
#
# if __name__ == "__main__":
#     main()
