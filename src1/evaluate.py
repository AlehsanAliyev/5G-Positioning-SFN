# import os
# import torch
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import geopandas as gpd
# from shapely.geometry import Point
# from sklearn.metrics import mean_squared_error
#
# from feature_engineering import merge_signal_data, extract_features
# from models.structured_mlp import StructuredMLP
# from models.fusion import UncertaintyWeightedFusion
#
# CHECKPOINT_DIR = "outputs/checkpoints/"
#
# def simulate_labels(n_samples):
#     lat_range = (41.104, 41.107)
#     lon_range = (29.025, 29.030)
#     x = np.random.uniform(lon_range[0], lon_range[1], n_samples)
#     y = np.random.uniform(lat_range[0], lat_range[1], n_samples)
#     return np.stack([x, y], axis=1)
#
# def load_models(input_dim):
#     models = []
#     for fname in os.listdir(CHECKPOINT_DIR):
#         if fname.endswith(".pth"):
#             model = StructuredMLP(input_dim)
#             model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, fname), map_location="cpu"))
#             model.eval()
#             models.append(model)
#     return models
#
# def plot_results_on_map(results_df):
#     print("🗺️ Plotting predictions on ITÜ map...")
#
#     shapefile_path = r"data/raw/maps/ITU_3DBINA_EPSG4326.shp"
#     gdf_buildings = gpd.read_file(shapefile_path)
#
#     # Create GeoDataFrames from predicted and true coordinates
#     true_points = [Point(xy) for xy in zip(results_df["true_lon"], results_df["true_lat"])]
#     pred_points = [Point(xy) for xy in zip(results_df["pred_lon"], results_df["pred_lat"])]
#
#     gdf_true = gpd.GeoDataFrame(geometry=true_points, crs="EPSG:4326")
#     gdf_pred = gpd.GeoDataFrame(geometry=pred_points, crs="EPSG:4326")
#
#     # Plot
#     fig, ax = plt.subplots(figsize=(10, 10))
#     gdf_buildings.plot(ax=ax, color="lightgray", edgecolor="black")
#     gdf_true.plot(ax=ax, color="blue", markersize=40, label="True")
#     gdf_pred.plot(ax=ax, color="red", markersize=40, label="Predicted")
#
#     plt.title("📍 Predicted vs True Positions on ITÜ Campus")
#     plt.xlabel("Longitude")
#     plt.ylabel("Latitude")
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#
#     os.makedirs("outputs/plots", exist_ok=True)
#     plot_path = "outputs/plots/map_prediction_overlay.png"
#     plt.savefig(plot_path)
#     print(f"🖼️ Saved map overlay: {plot_path}")
#     plt.show()
#
# def evaluate():
#     print("📥 Loading features...")
#     merged = merge_signal_data(stat_filter="Mean")
#     X_df = extract_features(merged)
#     X = X_df.fillna(X_df.median()).values
#     y_true = simulate_labels(len(X))
#
#     print("🧠 Loading models...")
#     input_dim = X.shape[1]
#     models = load_models(input_dim)
#     if not models:
#         print("❌ No models found!")
#         return
#
#     all_preds = []
#     all_vars = []
#
#     for model in models:
#         with torch.no_grad():
#             coords, log_vars = model(torch.tensor(X, dtype=torch.float32))
#             all_preds.append(coords.unsqueeze(1))   # (B, 1, 2)
#             all_vars.append(log_vars.unsqueeze(1))  # (B, 1, 1)
#
#     coord_preds = torch.cat(all_preds, dim=1)  # (B, N, 2)
#     log_vars = torch.cat(all_vars, dim=1)      # (B, N, 1)
#
#     # Fuse predictions
#     fuser = UncertaintyWeightedFusion()
#     fused_coords = fuser(coord_preds, log_vars)
#     fused_np = fused_coords.detach().numpy()
#
#     # Evaluate
#     rmse = np.sqrt(mean_squared_error(y_true, fused_np))
#     print(f"\n✅ RMSE (synthetic): {rmse:.4f} units")
#
#     # Save CSV
#     results_df = pd.DataFrame({
#         "true_lon": y_true[:, 0],
#         "true_lat": y_true[:, 1],
#         "pred_lon": fused_np[:, 0],
#         "pred_lat": fused_np[:, 1]
#     })
#
#     os.makedirs("outputs/results", exist_ok=True)
#     csv_path = "outputs/results/fused_predictions.csv"
#     results_df.to_csv(csv_path, index=False)
#     print(f"💾 Results saved to: {csv_path}")
#
#     # Plot map overlay
#     plot_results_on_map(results_df)
#
# if __name__ == "__main__":
#     evaluate()

#
# import os
# import torch
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import geopandas as gpd
# from shapely.geometry import Point
# from sklearn.metrics import mean_squared_error
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
# def plot_results_on_map(results_df):
#     shapefile_path = r"data/raw/maps/ITU_3DBINA_EPSG4326.shp"
#     gdf_buildings = gpd.read_file(shapefile_path)
#
#     true_pts = [Point(xy) for xy in zip(results_df["true_lon"], results_df["true_lat"])]
#     pred_pts = [Point(xy) for xy in zip(results_df["pred_lon"], results_df["pred_lat"])]
#
#     gdf_true = gpd.GeoDataFrame(geometry=true_pts, crs="EPSG:4326")
#     gdf_pred = gpd.GeoDataFrame(geometry=pred_pts, crs="EPSG:4326")
#
#     fig, ax = plt.subplots(figsize=(10, 10))
#     gdf_buildings.plot(ax=ax, color="lightgray", edgecolor="black")
#     gdf_true.plot(ax=ax, color="blue", label="True", markersize=40, alpha=0.6)
#     gdf_pred.plot(ax=ax, color="red", label="Predicted", markersize=40, alpha=0.6)
#
#     plt.title("📍 Predicted vs True GPS Locations (ITÜ Campus)")
#     plt.xlabel("Longitude")
#     plt.ylabel("Latitude")
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#
#     os.makedirs("outputs/plots", exist_ok=True)
#     plt.savefig("outputs/plots/map_prediction_overlay.png")
#     print("🖼️ Saved map overlay to outputs/plots/map_prediction_overlay.png")
#     plt.show()
#
# def evaluate():
#     print("📊 Loading real test data...")
#     df = merge_signal_data()
#     X, y_true = extract_features_and_labels(df)
#
#     print("📦 Loading trained model...")
#     model = load_model(input_dim=X.shape[1])
#
#     with torch.no_grad():
#         coords_pred, _ = model(torch.tensor(X, dtype=torch.float32))
#         coords_pred_np = coords_pred.numpy()
#
#     rmse = np.sqrt(mean_squared_error(y_true, coords_pred_np))
#     print(f"✅ RMSE on real GPS data: {rmse:.4f} degrees")
#
#     # Save predictions
#     results_df = pd.DataFrame({
#         "true_lon": y_true[:, 0],
#         "true_lat": y_true[:, 1],
#         "pred_lon": coords_pred_np[:, 0],
#         "pred_lat": coords_pred_np[:, 1],
#     })
#
#     os.makedirs("outputs/results", exist_ok=True)
#     results_df.to_csv("outputs/results/fused_predictions.csv", index=False)
#     print("💾 Saved prediction results to outputs/results/fused_predictions.csv")
#
#     # Plot map overlay
#     plot_results_on_map(results_df)
#
# if __name__ == "__main__":
#     evaluate()
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
from sklearn.metrics import mean_squared_error
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

def plot_results_on_map(results_df):
    shapefile_path = r"data/raw/maps/ITU_3DBINA_EPSG4326.shp"
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

    os.makedirs("outputs/plots", exist_ok=True)
    plt.savefig("outputs/plots/map_prediction_overlay.png")
    print("🖼️ Saved map overlay to outputs/plots/map_prediction_overlay.png")
    plt.show()

def evaluate():
    print("📊 Loading real test data...")
    df = merge_signal_data()
    X, y_true = extract_features_and_labels(df)

    print("🧠 Loading model and scaler...")
    model = load_model(input_dim=X.shape[1])
    y_scaler = joblib.load(SCALER_PATH)

    with torch.no_grad():
        coords_pred, _ = model(torch.tensor(X, dtype=torch.float32))
        coords_pred_np = coords_pred.numpy()

    # Unscale prediction
    coords_pred_inv = y_scaler.inverse_transform(coords_pred_np)

    rmse = np.sqrt(mean_squared_error(y_true, coords_pred_inv))
    print(f"✅ RMSE on real GPS data: {rmse:.4f} degrees")

    # Save predictions
    results_df = pd.DataFrame({
        "true_lon": y_true[:, 0],
        "true_lat": y_true[:, 1],
        "pred_lon": coords_pred_inv[:, 0],
        "pred_lat": coords_pred_inv[:, 1],
    })

    os.makedirs("outputs/results", exist_ok=True)
    results_df.to_csv("outputs/results/fused_predictions.csv", index=False)
    print("💾 Saved prediction results to outputs/results/fused_predictions.csv")

    # Map overlay
    plot_results_on_map(results_df)

if __name__ == "__main__":
    evaluate()
