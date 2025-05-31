import os
import glob
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import joblib
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point

from feature_engineering2 import merge_signal_data, extract_features_and_labels
from models.structured_mlp import StructuredMLP

CHECKPOINT_PATH = "outputs/checkpoints_v2/mlp_real_gps_scaled_v2.pth"
SCALER_PATH = "outputs/checkpoints_v2/y_scaler_v2.pkl"
RESULTS_DIR = "outputs/results_v2"
PLOTS_DIR = "outputs/plots_v2"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


def load_model(input_dim):
    model = StructuredMLP(input_dim)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location="cpu"))
    model.eval()
    return model


def plot_results_on_map(results_df):
    # Try to find shapefile case-insensitively
    matches = glob.glob("data/raw/maps/ITU_3DBINA_EPSG4326.*[sS][hH][pP]")
    if not matches:
        print("⚠️ Map shapefile not found. Skipping map plot.")
        return

    shapefile_path = matches[0]
    gdf_buildings = gpd.read_file(shapefile_path)

    true_pts = [Point(xy) for xy in zip(results_df["true_lon"], results_df["true_lat"])]
    pred_pts = [Point(xy) for xy in zip(results_df["pred_lon"], results_df["pred_lat"])]

    gdf_true = gpd.GeoDataFrame(geometry=true_pts, crs="EPSG:4326")
    gdf_pred = gpd.GeoDataFrame(geometry=pred_pts, crs="EPSG:4326")

    fig, ax = plt.subplots(figsize=(10, 10))
    gdf_buildings.plot(ax=ax, color="lightgray", edgecolor="black")
    gdf_true.plot(ax=ax, color="blue", label="True", markersize=40, alpha=0.6)
    gdf_pred.plot(ax=ax, color="red", label="Predicted", markersize=40, alpha=0.6)

    plt.title("📍 Predicted vs True GPS Locations (ITÜ Campus)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "map_prediction_overlay_v2.png"))
    plt.show()


def degrees_to_meters(rmse_deg):
    return rmse_deg * 111_320  # Approx. conversion factor for ITÜ region


def evaluate():
    print("📊 Loading test data...")
    df = merge_signal_data()
    X, y_true, _ = extract_features_and_labels(df)

    print("🧠 Loading model and scaler...")
    model = load_model(input_dim=X.shape[1])
    y_scaler = joblib.load(SCALER_PATH)

    with torch.no_grad():
        coords_pred, _ = model(torch.tensor(X, dtype=torch.float32))
        coords_pred_np = coords_pred.numpy()

    coords_pred_inv = y_scaler.inverse_transform(coords_pred_np)

    rmse = np.sqrt(mean_squared_error(y_true, coords_pred_inv))
    rmse_m = degrees_to_meters(rmse)
    print(f"✅ RMSE on test set: {rmse:.4f} degrees (~{rmse_m:.2f} meters)")

    results_df = pd.DataFrame({
        "true_lon": y_true[:, 0],
        "true_lat": y_true[:, 1],
        "pred_lon": coords_pred_inv[:, 0],
        "pred_lat": coords_pred_inv[:, 1],
    })

    results_df.to_csv(os.path.join(RESULTS_DIR, "fused_predictions_v2.csv"), index=False)
    print("💾 Saved prediction results.")

    plot_results_on_map(results_df)


if __name__ == "__main__":
    evaluate()