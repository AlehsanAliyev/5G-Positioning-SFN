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

from dataloader2 import load_base_station_config, load_uplink_series_data, load_downlink_series_data
from feature_engineering2 import merge_signal_data, extract_features_and_labels
from models.structured_mlp import StructuredMLP

CHECKPOINT_PATH = "outputs/checkpoints_v2/mlp_real_gps_scaled_v2.pth"
SCALER_PATH = "outputs/checkpoints_v2/y_scaler_v2.pkl"

DL_CHECKPOINT = "outputs/checkpoints_v2/mlp_real_gps_scaled_v2_dl.pth"
UL_CHECKPOINT = "outputs/checkpoints_v2/mlp_real_gps_scaled_v2_ul.pth"
# SCAN_CHECKPOINT = "outputs/checkpoints_v2/mlp_real_gps_scaled_v2_scanner.pth"

DL_SCALER = "outputs/checkpoints_v2/y_scaler_v2_dl.pkl"
UL_SCALER = "outputs/checkpoints_v2/y_scaler_v2_ul.pkl"
# SCAN_SCALER = "outputs/checkpoints_v2/y_scaler_v2_scanner.pkl"

RESULTS_DIR = "outputs/results_v2"
PLOTS_DIR = "outputs/plots_v2"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


def load_model_and_scaler(path_model, path_scaler, input_dim):
    model = StructuredMLP(input_dim)
    model.load_state_dict(torch.load(path_model, map_location="cpu"))
    model.eval()
    scaler = joblib.load(path_scaler)
    return model, scaler

def predict(model, X):
    with torch.no_grad():
        coords_pred, _ = model(torch.tensor(X, dtype=torch.float32))
    return coords_pred.numpy()

def fuse_predictions(*preds, weights=None):
    preds = np.stack(preds)
    if weights is None:
        weights = np.ones(len(preds))
    weights = np.array(weights).reshape(-1, 1, 1)
    fused = np.sum(preds * weights, axis=0) / np.sum(weights)
    return fused



def plot_results_on_map(results_df):
    # Try to find shapefile case-insensitively
    matches = glob.glob("data/raw/maps/ITU_3DBINA_EPSG4326.*[sS][hH][pP]")
    if not matches:
        print("⚠️ Map shapefile not found. Skipping map plot.")
        return

    df = load_base_station_config()

    shapefile_path = matches[0]
    gdf_buildings = gpd.read_file(shapefile_path)

    true_pts = [Point(xy) for xy in zip(results_df["true_lon"], results_df["true_lat"])]
    pred_pts = [Point(xy) for xy in zip(results_df["pred_lon"], results_df["pred_lat"])]
    tower_pts = [Point(xy) for xy in zip(df["lon"], df["lat"])]

    gdf_true = gpd.GeoDataFrame(geometry=true_pts, crs="EPSG:4326")
    gdf_pred = gpd.GeoDataFrame(geometry=pred_pts, crs="EPSG:4326")
    tower_pts = gpd.GeoDataFrame(geometry=tower_pts, crs="EPSG:4326")

    fig, ax = plt.subplots(figsize=(10, 10))
    gdf_buildings.plot(ax=ax, color="lightgray", edgecolor="black")
    gdf_true.plot(ax=ax, color="blue", label="True", markersize=4, alpha=0.6)
    gdf_pred.plot(ax=ax, color="red", label="Predicted", markersize=4, alpha=0.6)
    tower_pts.plot(ax=ax, color="green", label="Tower", markersize=30, alpha=0.6)

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


# Define column-specific invalid thresholds
COLUMN_THRESHOLDS = {
    "RSRP": -150,
    "RSRQ": -30,
    "SINR": -10,
}

# Determine which threshold applies to each column
def get_threshold(colname):
    return -9999 # No threshold applied
    if "RSRP" in colname:
        return COLUMN_THRESHOLDS["RSRP"]
    elif "RSRQ" in colname:
        return COLUMN_THRESHOLDS["RSRQ"]
    elif "SINR" in colname:
        return COLUMN_THRESHOLDS["SINR"]
    else:
        return -9999  # default catch-all threshold

# Masked mean that accepts a threshold
def masked_mean(series, threshold):
    valid = series[series > threshold]
    return valid.mean() if not valid.empty else np.nan

# Modified function to return two separate aligned DataFrames for UL and DL models

def align_dl_ul_separately(df_dl, df_ul, precision=5):
    def prepare(df, prefix):
        df = df.copy()
        df['loc_key'] = df["Latitude"].round(precision).astype(str) + "_" + df["Longitude"].round(precision).astype(str)
        signal_cols = [col for col in df.columns if col not in ["Latitude", "Longitude", "loc_key"]]
        
        agg_funcs = {
            col: (lambda x, col=col: masked_mean(x, get_threshold(col)))
            for col in signal_cols
        }

        grouped = df.groupby("loc_key").agg(agg_funcs)
        grouped.columns = [f"{prefix}_{col}" for col in grouped.columns]
        return grouped

    dl_agg = prepare(df_dl, "DL")
    ul_agg = prepare(df_ul, "UL")

    # Get common location keys
    common_keys = dl_agg.index.intersection(ul_agg.index)

    # Filter both DataFrames to only include common keys
    dl_common = dl_agg.loc[common_keys]
    ul_common = ul_agg.loc[common_keys]

    # Reattach coordinates
    latlon = df_dl.copy()
    latlon['loc_key'] = latlon["Latitude"].round(precision).astype(str) + "_" + latlon["Longitude"].round(precision).astype(str)
    latlon = latlon.drop_duplicates('loc_key')[['loc_key', 'Latitude', 'Longitude']].set_index('loc_key')
    latlon = latlon.loc[common_keys]

    df_dl_final = pd.concat([latlon, dl_common], axis=1).reset_index(drop=True)
    df_ul_final = pd.concat([latlon, ul_common], axis=1).reset_index(drop=True)

    return df_dl_final, df_ul_final


def evaluate():
    print("📊 Loading fused dataset...")
    df_ul = load_uplink_series_data()
    df_ul = merge_signal_data(df_ul)
    
    df_dl = load_downlink_series_data()
    df_dl = merge_signal_data(df_dl)

    df_dl, df_ul = align_dl_ul_separately(df_dl, df_ul)
    df_ul.to_csv("analysis_outputs/ul_alligned.csv", index=False)
    df_dl.to_csv("analysis_outputs/dl_alligned.csv", index=False)
        

    X_ul, y, _ = extract_features_and_labels(df_ul)
    X_dl, y, _ = extract_features_and_labels(df_dl)

    print("🧠 Loading models and scalers...")
    dl_model, dl_scaler = load_model_and_scaler(DL_CHECKPOINT, DL_SCALER, input_dim=X_dl.shape[1])
    ul_model, ul_scaler = load_model_and_scaler(UL_CHECKPOINT, UL_SCALER, input_dim=X_ul.shape[1])
    # scan_model, scan_scaler = load_model_and_scaler(SCAN_CHECKPOINT, SCAN_SCALER, input_dim=X.shape[1])

    print("🔮 Predicting...")
    pred_dl_scaled = predict(dl_model, X_dl)
    pred_ul_scaled = predict(ul_model, X_ul)
    # pred_sc_scaled = predict(scan_model, X)

    pred_dl = dl_scaler.inverse_transform(pred_dl_scaled)
    pred_ul = ul_scaler.inverse_transform(pred_ul_scaled)
    # pred_sc = scan_scaler.inverse_transform(pred_sc_scaled)

    print("🔗 Fusing predictions...")
    fused_preds = fuse_predictions(pred_dl, pred_ul, weights=[1, 1])

    rmse = np.sqrt(mean_squared_error(y, fused_preds))
    rmse_m = degrees_to_meters(rmse)
    print(f"✅ Fused RMSE: {rmse:.4f} degrees (~{rmse_m:.2f} meters)")

    results_df = pd.DataFrame({
        "true_lon": y[:, 0],
        "true_lat": y[:, 1],
        "pred_lon": fused_preds[:, 0],
        "pred_lat": fused_preds[:, 1],
    })

    results_df.to_csv(os.path.join(RESULTS_DIR, "fused_predictions.csv"), index=False)
    print("💾 Saved fused prediction results.")

    plot_results_on_map(results_df)

if __name__ == "__main__":
    evaluate()