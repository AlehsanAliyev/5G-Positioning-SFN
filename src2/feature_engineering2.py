import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from dataloader2 import load_base_station_config

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


def merge_signal_data(df) -> pd.DataFrame:
    df = df.dropna(subset=["Latitude", "Longitude"])
    nunique = df.nunique()
    df = df.drop(columns=nunique[nunique <= 1].index)

    # Fill RSRP and RSRQ fields with domain-aware defaults
    fill_map = {
        col: -200 for col in df.columns if 'RSRP' in col
    }
    fill_map.update({
        col: -30 for col in df.columns if 'RSRQ' in col
    })
    fill_map.update({
        col: 0 for col in df.columns if 'PCI' in col
    })

    df = df.fillna(value=fill_map)

    coord = load_base_station_config()
    pci_columns = ['NR_UE_PCI_0', 'NR_UE_Nbr_PCI_0', 'NR_UE_Nbr_PCI_1', 'NR_UE_Nbr_PCI_2', 'NR_UE_Nbr_PCI_3']
    
    for pci_col in pci_columns:
        if pci_col not in df.columns:
            continue

        df[pci_col] = df[pci_col].astype(int)
        merge_df = coord.rename(columns={
            'lat': f'{pci_col}_lat',
            'lon': f'{pci_col}_lon',
            'azimuth': f'{pci_col}_azimuth',
            'height': f'{pci_col}_height',
            'pci': pci_col
        })

        df = df.merge(merge_df, how='left', on=pci_col)

    fill_map = {
        col: 0 for col in df.columns if 'lat' in col
    }
    fill_map.update({
        col: 0 for col in df.columns if 'lon' in col
    })
    fill_map.update({
        col: 0 for col in df.columns if 'azimuth' in col
    })
    fill_map.update({
        col: 0 for col in df.columns if 'height' in col
    })
    df = df.fillna(value=fill_map)

    return df.reset_index(drop=True)


def extract_features_and_labels(df: pd.DataFrame, drop_threshold=0.5, scale=True):
    """
    Extracts cleaned and relevant features + GPS labels (lon, lat).
    Includes signal stats and engineered neighbor features.
    - Fills remaining NaNs with median
    - Scales features with StandardScaler
    """
    y = df[["Longitude", "Latitude"]].values

    # Keep only numeric data
    X = df.select_dtypes(include=[np.number]).drop(columns=["Longitude", "Latitude"])

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
    
if __name__ == "__main__":
    preview()
