import torch
import joblib
import numpy as np

from feature_engineering2 import merge_signal_data, extract_features_and_labels
from models.structured_mlp import StructuredMLP

CHECKPOINT_PATH = "outputs/checkpoints_v2/mlp_real_gps_scaled_v2.pth"
SCALER_PATH = "outputs/checkpoints_v2/y_scaler_v2.pkl"


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
    print("📥 Loading data for inference...")
    df = merge_signal_data()
    X, y_true, _ = extract_features_and_labels(df)

    print("🧠 Loading model and scaler...")
    model = load_model(input_dim=X.shape[1])
    y_scaler = joblib.load(SCALER_PATH)

    index = 0  # Change this to test a different row
    pred_scaled = predict_single_sample(model, X[index])
    pred_coords = y_scaler.inverse_transform(pred_scaled.reshape(1, -1)).squeeze()
    true_coords = y_true[index]

    print("\n📍 Inference Result:")
    print(f"Predicted → Lon: {pred_coords[0]:.6f}, Lat: {pred_coords[1]:.6f}")
    print(f"Ground Truth → Lon: {true_coords[0]:.6f}, Lat: {true_coords[1]:.6f}")


if __name__ == "__main__":
    main()