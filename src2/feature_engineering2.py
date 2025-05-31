import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from dataloader2 import load_downlink_series_data

# Final selected signal-related features based on domain and data analysis
RELEVANT_FEATURES = [
    "NR_UE_RSRP_0",
    "NR_UE_RSRQ_0",
    "NR_UE_SINR_0",
    "NR_UE_Timing_Advance",
    "NR_UE_Pathloss_DL_0",
    "NR_UE_Modulation_Avg_DL_0",
    "App_Throughput_DL"
]

# Neighbor cell signal features
NEIGHBOR_RSRP_COLS = [f"NR_UE_Nbr_RSRP_{i}" for i in range(5)]
NEIGHBOR_SINR_COLS = [f"NR_UE_Nbr_SINR_{i}" for i in range(5)]


def merge_signal_data() -> pd.DataFrame:
    df = load_downlink_series_data()
    df = df.dropna(subset=["Latitude", "Longitude"])
    nunique = df.nunique()
    df = df.drop(columns=nunique[nunique <= 1].index)
    return df.reset_index(drop=True)


def extract_features_and_labels(df: pd.DataFrame, drop_threshold=0.5, scale=True):
    """
    Extracts cleaned and relevant features + GPS labels (lon, lat).
    Includes signal stats and engineered neighbor features.
    - Drops columns with > drop_threshold missing
    - Fills remaining NaNs with median
    - Scales features with StandardScaler
    """
    y = df[["Longitude", "Latitude"]].values

    # Select columns below missingness threshold
    null_fraction = df.isnull().mean()
    keep_cols = null_fraction[null_fraction < drop_threshold].index.tolist()

    # Remove position columns from features if present
    for col in ["Latitude", "Longitude"]:
        if col in keep_cols:
            keep_cols.remove(col)

    # Keep only numeric data
    X = df[keep_cols].select_dtypes(include=[np.number])

    # Fill NaNs with median
    X = X.fillna(X.median(numeric_only=True))

    if X.empty:
        print("⚠️ No valid rows after filtering. Returning empty arrays.")
        return np.empty((0, len(X.columns))), np.empty((0, 2)), None

    if scale:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled, y, scaler
    else:
        return X.values, y, None


def preview():
    print("\n🔍 Previewing engineered features...")
    df = merge_signal_data()
    print("🧾 Raw shape:", df.shape)
    X, y, _ = extract_features_and_labels(df)
    print("✅ Features shape:", X.shape)
    print("✅ Labels shape:", y.shape)
    if len(X) > 0:
        print("📈 First 3 feature vectors:", X[:3])
        print("📍 First 3 GPS coords:", y[:3])
    else:
        print("❌ No samples to preview.")

if __name__ == "__main__":
    preview()
