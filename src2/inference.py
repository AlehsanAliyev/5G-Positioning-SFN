from pathlib import Path
import torch
import joblib

from feature_engineering2 import build_dataset, extract_features_and_labels
from models.structured_mlp import StructuredMLP

BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = BASE_DIR / "outputs" / "checkpoints"
CHECKPOINT_PATH = CHECKPOINT_DIR / "mlp_dl_geo.pth"
X_SCALER_PATH = CHECKPOINT_DIR / "x_scaler_dl_geo.pkl"
Y_SCALER_PATH = CHECKPOINT_DIR / "y_scaler_dl_geo.pkl"


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
    print("Loading data for inference...")
    df = build_dataset()

    x_scaler = joblib.load(X_SCALER_PATH)
    X, y_true, _ = extract_features_and_labels(df, scale=True, scaler=x_scaler)

    print("Loading model and scaler...")
    model = load_model(input_dim=X.shape[1])
    y_scaler = joblib.load(Y_SCALER_PATH)

    index = 0
    pred_scaled = predict_single_sample(model, X[index])
    pred_coords = y_scaler.inverse_transform(pred_scaled.reshape(1, -1)).squeeze()
    true_coords = y_true[index]

    print("")
    print("Inference Result:")
    print(f"Predicted Lon: {pred_coords[0]:.6f}, Lat: {pred_coords[1]:.6f}")
    print(f"Ground Truth Lon: {true_coords[0]:.6f}, Lat: {true_coords[1]:.6f}")


if __name__ == "__main__":
    main()
