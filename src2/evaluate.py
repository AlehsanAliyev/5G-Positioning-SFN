from pathlib import Path
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import joblib
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point

from feature_engineering2 import build_dataset, extract_features_and_labels
from models.structured_mlp import StructuredMLP

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
OUTPUT_DIR = BASE_DIR / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
RESULTS_DIR = OUTPUT_DIR / "results"
PLOTS_DIR = OUTPUT_DIR / "plots"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_PATH = CHECKPOINT_DIR / "mlp_dl_geo.pth"
X_SCALER_PATH = CHECKPOINT_DIR / "x_scaler_dl_geo.pkl"
Y_SCALER_PATH = CHECKPOINT_DIR / "y_scaler_dl_geo.pkl"

EARTH_RADIUS_M = 6_371_000


def load_model(input_dim):
    model = StructuredMLP(input_dim)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location="cpu"))
    model.eval()
    return model


def vectorized_haversine(lat1, lon1, lat2, lon2):
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return EARTH_RADIUS_M * c


def summarize_errors(results_df):
    errors_m = vectorized_haversine(
        results_df["true_lat"].values,
        results_df["true_lon"].values,
        results_df["pred_lat"].values,
        results_df["pred_lon"].values,
    )
    rmse = np.sqrt(np.mean(errors_m ** 2))

    print(f"Haversine RMSE: {rmse:.2f} m")
    print(f"Mean Error: {np.mean(errors_m):.2f} m")
    print(f"Median Error: {np.median(errors_m):.2f} m")
    print(f"Max Error: {np.max(errors_m):.2f} m")
    print(f"P90 Error: {np.percentile(errors_m, 90):.2f} m")
    print(f"P95 Error: {np.percentile(errors_m, 95):.2f} m")

    results_df = results_df.copy()
    results_df["haversine_error_m"] = errors_m
    results_df.to_csv(RESULTS_DIR / "prediction_errors.csv", index=False)
    print("Saved error details.")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(errors_m, bins=50, color="steelblue", edgecolor="black")
    ax.set_title("Haversine Error Distribution")
    ax.set_xlabel("Error (m)")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "error_histogram.png")
    plt.show()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot(errors_m, vert=False)
    ax.set_title("Haversine Error Boxplot")
    ax.set_xlabel("Error (m)")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "error_boxplot.png")
    plt.show()


def plot_results_on_map(results_df, max_points=5000):
    shapefile_path = PROJECT_DIR / "data" / "raw" / "maps" / "ITU_3DBINA_EPSG4326.shp"
    if not shapefile_path.exists():
        print("Map shapefile not found. Skipping map plot.")
        return

    plot_df = results_df
    if len(results_df) > max_points:
        plot_df = results_df.sample(n=max_points, random_state=42).reset_index(drop=True)

    padding = 0.0005
    min_lon = plot_df["true_lon"].min() - padding
    max_lon = plot_df["true_lon"].max() + padding
    min_lat = plot_df["true_lat"].min() - padding
    max_lat = plot_df["true_lat"].max() + padding

    bbox = (min_lon, min_lat, max_lon, max_lat)
    gdf_buildings = gpd.read_file(shapefile_path, bbox=bbox)

    true_pts = [Point(xy) for xy in zip(plot_df["true_lon"], plot_df["true_lat"])]
    pred_pts = [Point(xy) for xy in zip(plot_df["pred_lon"], plot_df["pred_lat"])]

    gdf_true = gpd.GeoDataFrame(geometry=true_pts, crs="EPSG:4326")
    gdf_pred = gpd.GeoDataFrame(geometry=pred_pts, crs="EPSG:4326")

    fig, ax = plt.subplots(figsize=(10, 10))
    gdf_buildings.plot(ax=ax, color="lightgray", edgecolor="black")
    gdf_true.plot(ax=ax, color="blue", label="True", markersize=40, alpha=0.6)
    gdf_pred.plot(ax=ax, color="red", label="Predicted", markersize=40, alpha=0.6)

    plt.title("Predicted vs True GPS Locations (ITU Campus)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "map_prediction_overlay.png")
    plt.show()


def evaluate():
    print("Loading test data...")
    df = build_dataset()

    x_scaler = joblib.load(X_SCALER_PATH)
    X, y_true, _ = extract_features_and_labels(df, scale=True, scaler=x_scaler)

    print("Loading model and scaler...")
    model = load_model(input_dim=X.shape[1])
    y_scaler = joblib.load(Y_SCALER_PATH)

    with torch.no_grad():
        coords_pred, _ = model(torch.tensor(X, dtype=torch.float32))
        coords_pred_np = coords_pred.cpu().numpy()

    coords_pred_inv = y_scaler.inverse_transform(coords_pred_np)

    rmse = np.sqrt(mean_squared_error(y_true, coords_pred_inv))
    rmse_m = rmse * 111_320
    print(f"RMSE on test set: {rmse:.4f} degrees (~{rmse_m:.2f} meters)")

    results_df = pd.DataFrame({
        "true_lon": y_true[:, 0],
        "true_lat": y_true[:, 1],
        "pred_lon": coords_pred_inv[:, 0],
        "pred_lat": coords_pred_inv[:, 1],
    })

    results_df.to_csv(RESULTS_DIR / "dl_predictions.csv", index=False)
    print("Saved prediction results.")

    summarize_errors(results_df)
    plot_results_on_map(results_df)


if __name__ == "__main__":
    evaluate()
