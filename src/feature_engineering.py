# '''
# ✅ Merge Downlink, Uplink, and Scanner data by index (since you only have 1 "Mean" row for now)
# ✅ Add location (lat, lon) from DL as the ground truth
# ✅ Prepare per-base-station feature vectors (for Structured MLPs)
# ✅ (Later: integrate base station geometry or map info)
# '''
# import pandas as pd
# from data_loader import (
#     load_downlink_data,
#     load_uplink_data,
#     load_scanner_data
# )
#
# def merge_signal_data(stat_filter=None) -> pd.DataFrame:
#     """
#     Merge Downlink, Uplink, and Scanner data by index.
#     You can specify a `stat_filter` like "Mean" or None to load all rows.
#     """
#     dl = load_downlink_data(stat_filter=stat_filter)
#     ul = load_uplink_data(stat_filter=stat_filter)
#     scanner = load_scanner_data(stat_filter=stat_filter)
#
#     print("📄 DL Columns:", dl.columns.tolist())
#
#     # Add prefixes to avoid name collisions
#     dl = dl.add_prefix("DL_")
#     ul = ul.add_prefix("UL_")
#     scanner = scanner.add_prefix("SCN_")
#
#     # Merge by index
#     merged = pd.concat([dl, ul, scanner], axis=1)
#     return merged
#
# def extract_features(merged: pd.DataFrame) -> pd.DataFrame:
#     """
#     Drop non-feature columns (like Statistic fields).
#     Returns only numerical input features.
#     """
#     exclude = [col for col in merged.columns if "Statistic" in col]
#     features = merged.drop(columns=exclude, errors="ignore")
#     return features
#
# def preview():
#     merged = merge_signal_data(stat_filter=None)  # Load all stat rows
#     X = extract_features(merged)
#
#     print("✅ Merged shape:", merged.shape)
#     print("✅ Feature shape:", X.shape)
#     print("\n📌 Sample feature columns:", list(X.columns)[:10])
#     print("📌 Sample row:\n", X.iloc[0])
#
# if __name__ == "__main__":
#     preview()

import pandas as pd
from sklearn.preprocessing import StandardScaler
from data_loader import load_downlink_series_data

def merge_signal_data() -> pd.DataFrame:
    """
    Loads GPS-annotated downlink data.
    """
    df = load_downlink_series_data()

    # Drop rows with missing lat/lon
    df = df.dropna(subset=["lat", "lon"])
    return df.reset_index(drop=True)

def extract_features_and_labels(df: pd.DataFrame, drop_threshold=0.5):
    """
    Processes the data into X (features) and y (lon, lat) with smart missing value handling.
    - Drops columns with >50% missing
    - Fills remaining NaNs with median
    - Scales features with StandardScaler
    """
    # Separate labels
    y = df[["lon", "lat"]].values

    # Drop columns with > drop_threshold missing
    null_fraction = df.isnull().mean()
    keep_cols = null_fraction[null_fraction < drop_threshold].index.tolist()

    # Exclude position columns
    for col in ["lat", "lon"]:
        if col in keep_cols:
            keep_cols.remove(col)

    # Filter DataFrame to good columns
    X = df[keep_cols]

    # Impute remaining NaNs
    X = X.fillna(X.median(numeric_only=True))

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y

def preview():
    print("🔍 Previewing features and labels...")
    df = merge_signal_data()
    print("🧾 Raw shape:", df.shape)

    X, y = extract_features_and_labels(df)
    print("✅ Cleaned features shape:", X.shape)
    print("✅ Labels shape:", y.shape)
    print("📍 Sample label:", y[0])
    print("📈 Sample features:", X[0][:5])

if __name__ == "__main__":
    preview()
