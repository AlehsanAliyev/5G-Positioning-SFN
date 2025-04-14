# import os
# import torch
# import pandas as pd
# import numpy as np
# from models.structured_mlp import StructuredMLP
# from models.fusion import UncertaintyWeightedFusion
# from feature_engineering import merge_signal_data, extract_features
#
# CHECKPOINT_DIR = "outputs/checkpoints/"
#
# def load_mlp_model(path, input_dim):
#     model = StructuredMLP(input_dim)
#     model.load_state_dict(torch.load(path, map_location='cpu'))
#     model.eval()
#     return model
#
# def get_feature_vector():
#     # Load and prepare one-row input for test
#     merged = merge_signal_data(stat_filter="Mean")
#     X_df = extract_features(merged)
#
#     # Use only the first row for this demo
#     sample = X_df.iloc[[0]].fillna(X_df.median())
#     return sample
#
# def run_inference():
#     print("🧠 Loading models for inference...")
#     sample = get_feature_vector()
#     input_tensor = torch.tensor(sample.values, dtype=torch.float32)
#
#     coord_preds = []
#     log_vars = []
#
#     for fname in os.listdir(CHECKPOINT_DIR):
#         if not fname.endswith(".pth"):
#             continue
#
#         print(f"🔍 Using model: {fname}")
#         model = load_mlp_model(os.path.join(CHECKPOINT_DIR, fname), input_dim=input_tensor.shape[1])
#         with torch.no_grad():
#             coords, log_var = model(input_tensor)
#             coord_preds.append(coords)
#             log_vars.append(log_var)
#
#     if not coord_preds:
#         print("❌ No valid models found for inference.")
#         return
#
#     # Stack all predictions
#     coord_preds = torch.stack(coord_preds, dim=1)  # (B, N, 2)
#     log_vars = torch.stack(log_vars, dim=1)        # (B, N, 1)
#
#     # Fuse predictions
#     fuser = UncertaintyWeightedFusion()
#     fused_coords = fuser(coord_preds, log_vars)
#
#     fused_coords_np = fused_coords.squeeze().numpy()
#     print("\n📍 Final fused position prediction:")
#     print(f"Longitude: {fused_coords_np[0]:.6f}, Latitude: {fused_coords_np[1]:.6f}")
#
# if __name__ == "__main__":
#     run_inference()

#
#
# import torch
# import numpy as np
# import pandas as pd
#
# from models.structured_mlp import StructuredMLP
# from feature_engineering import merge_signal_data, extract_features_and_labels
#
# CHECKPOINT_PATH = "outputs/checkpoints/mlp_real_gps.pth"
#
# def load_model(input_dim):
#     model = StructuredMLP(input_dim)
#     model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location="cpu"))
#     model.eval()
#     return model
#
# def predict_single_sample(model, X_row):
#     with torch.no_grad():
#         coords_pred, _ = model(torch.tensor(X_row, dtype=torch.float32).unsqueeze(0))
#         return coords_pred.squeeze().numpy()
#
# def main():
#     print("📥 Loading real data...")
#     df = merge_signal_data()
#     X, y = extract_features_and_labels(df)
#
#     print("🧠 Loading trained model...")
#     model = load_model(input_dim=X.shape[1])
#
#     # Choose an index (e.g., 0 for first row)
#     index = 0
#     pred_coords = predict_single_sample(model, X[index])
#     true_coords = y[index]
#
#     print("\n📍 Inference Result:")
#     print(f"Predicted → Lon: {pred_coords[0]:.6f}, Lat: {pred_coords[1]:.6f}")
#     print(f"Ground Truth → Lon: {true_coords[0]:.6f}, Lat: {true_coords[1]:.6f}")
#
# if __name__ == "__main__":
#     main()


import torch
import numpy as np
import joblib

from models.structured_mlp import StructuredMLP
from feature_engineering import merge_signal_data, extract_features_and_labels

CHECKPOINT_PATH = "outputs/checkpoints/mlp_real_gps_scaled.pth"
SCALER_PATH = "outputs/checkpoints/y_scaler.pkl"

def load_model(input_dim):
    model = StructuredMLP(input_dim)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location="cpu"))
    model.eval()
    return model

def predict_single_sample(model, X_row):
    with torch.no_grad():
        coords_pred, _ = model(torch.tensor(X_row, dtype=torch.float32).unsqueeze(0))
        return coords_pred.squeeze().numpy()

def main():
    print("📥 Loading real data...")
    df = merge_signal_data()
    X, y_true = extract_features_and_labels(df)

    print("🧠 Loading model and label scaler...")
    model = load_model(input_dim=X.shape[1])
    y_scaler = joblib.load(SCALER_PATH)

    index = 0  # You can change this to test other rows
    pred_scaled = predict_single_sample(model, X[index])
    pred_coords = y_scaler.inverse_transform(pred_scaled.reshape(1, -1)).squeeze()
    true_coords = y_true[index]

    print("\n📍 Inference Result:")
    print(f"Predicted → Lon: {pred_coords[0]:.6f}, Lat: {pred_coords[1]:.6f}")
    print(f"Ground Truth → Lon: {true_coords[0]:.6f}, Lat: {true_coords[1]:.6f}")

if __name__ == "__main__":
    main()
