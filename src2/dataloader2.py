from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
RAW_DATA_DIRS = [
    PROJECT_DIR / "data" / "raw",
    PROJECT_DIR / "data" / "Sample_Data_2",
]
OUTPUT_DIR = BASE_DIR / "analysis_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DL_FEATURES = [
    "NR_UE_Throughput_PDCP_DL", "NR_UE_PCI_0", "NR_UE_RB_Num_DL_0",
    "NR_UE_Pathloss_DL_0", "App_Throughput_DL", "NR_UE_Power_Tx_PUSCH_0",
    "NR_UE_RSRP_0", "NR_UE_BLER_DL_0", "NR_UE_NACK_Rate_DL_0",
    "NR_UE_NACK_Rate_UL_0", "NR_UE_Ack_As_Nack_DL_0", "NR_UE_Nbr_RSRP_0",
    "NR_UE_Nbr_PCI_0", "NR_UE_SINR_0", "NR_UE_RSRQ_0", "NR_UE_Nbr_RSRQ_0",
    "NR_UE_Nbr_PCI_1", "NR_UE_Nbr_PCI_2", "NR_UE_Timing_Advance",
    "NR_UE_Nbr_RSRP_1", "NR_UE_Nbr_RSRP_2",
]

SIGNAL_COLS = [
    "NR_UE_RSRP_0", "NR_UE_SINR_0", "NR_UE_RSRQ_0",
    "NR_UE_Throughput_PDCP_DL", "NR_UE_Pathloss_DL_0", "App_Throughput_DL",
]

PCI_BASES = ["NR_UE_PCI_0", "NR_UE_Nbr_PCI_0", "NR_UE_Nbr_PCI_1", "NR_UE_Nbr_PCI_2"]
EARTH_RADIUS_M = 6_371_000


def find_optional_file(filename: str):
    for base in RAW_DATA_DIRS:
        path = base / filename
        if path.exists():
            return path
    return None


def require_files(filenames):
    paths = []
    missing = []
    for filename in filenames:
        path = find_optional_file(filename)
        if path is None:
            missing.append(filename)
        else:
            paths.append(path)
    if not paths:
        raise FileNotFoundError("No input files found: " + ", ".join(missing))
    if missing:
        print("Warning: missing files:", ", ".join(missing))
    return paths


def find_base_station_config_file():
    for base in RAW_DATA_DIRS:
        if not base.exists():
            continue
        for path in base.glob("*.xlsx"):
            if "bilgileri" in path.name.lower():
                return path
    return None


def load_base_station_config():
    path = find_base_station_config_file()
    if path is None:
        raise FileNotFoundError("Base station config .xlsx not found in raw data folders.")

    xl = pd.ExcelFile(path)
    sheet_name = next((s for s in xl.sheet_names if "tablosu" in s.lower()), xl.sheet_names[0])
    df = xl.parse(sheet_name=sheet_name)

    col_map = {}
    for col in df.columns:
        lower = str(col).lower()
        if "latitude" in lower:
            col_map[col] = "lat"
        elif "longitude" in lower:
            col_map[col] = "lon"
        elif "azimuth" in lower:
            col_map[col] = "azimuth"
        elif "height" in lower:
            col_map[col] = "height"
        elif "pci" in lower:
            col_map[col] = "pci"
    df = df.rename(columns=col_map)

    required = ["lat", "lon", "azimuth", "height", "pci"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("Missing required columns in base station config: " + ", ".join(missing))

    return df[required]


def haversine_distance(lat1, lon1, lat2, lon2):
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return EARTH_RADIUS_M * c


def bearing_from_bs_to_ue(bs_lat, bs_lon, ue_lat, ue_lon):
    lat1 = np.radians(bs_lat)
    lon1 = np.radians(bs_lon)
    lat2 = np.radians(ue_lat)
    lon2 = np.radians(ue_lon)
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360


def apply_geometry_features(df, pci_col_base):
    bs_lat = df[f"{pci_col_base}_lat"]
    bs_lon = df[f"{pci_col_base}_lon"]
    bs_azimuth = df[f"{pci_col_base}_azimuth"]
    ue_lat = df["Latitude"]
    ue_lon = df["Longitude"]

    df[f"{pci_col_base}_distance"] = haversine_distance(bs_lat, bs_lon, ue_lat, ue_lon)
    df[f"{pci_col_base}_bearing"] = bearing_from_bs_to_ue(bs_lat, bs_lon, ue_lat, ue_lon)
    delta_angle = np.abs(df[f"{pci_col_base}_bearing"] - bs_azimuth)
    df[f"{pci_col_base}_angle_offset"] = np.minimum(delta_angle, 360 - delta_angle)
    df[f"{pci_col_base}_cos_offset"] = np.cos(np.radians(df[f"{pci_col_base}_angle_offset"]))

    rsrp_col = f"{pci_col_base.replace('PCI', 'RSRP')}"
    if rsrp_col in df.columns:
        df[f"{pci_col_base}_rsrp_weighted"] = df[rsrp_col] * df[f"{pci_col_base}_cos_offset"]

    return df


def is_useless_signal(row):
    critical_cols = [
        "NR_UE_RSRP_0", "NR_UE_SINR_0", "NR_UE_RSRQ_0",
        "NR_UE_Nbr_RSRP_0", "NR_UE_Nbr_RSRP_1", "NR_UE_Nbr_RSRP_2",
    ]
    if row[critical_cols].isna().all():
        return True
    if pd.notna(row.get("NR_UE_RSRP_0")) and pd.notna(row.get("NR_UE_SINR_0")):
        if (row["NR_UE_RSRP_0"] < -125) and (row["NR_UE_SINR_0"] < -5):
            return True
    return False


def enrich_with_base_stations(df):
    coord = load_base_station_config()
    pci_columns = [col for col in df.columns if "PCI" in col]

    for pci_col in pci_columns:
        df[pci_col] = pd.to_numeric(df[pci_col], errors="coerce").astype("Int64")

        merge_df = coord.rename(columns={
            "lat": f"{pci_col}_lat",
            "lon": f"{pci_col}_lon",
            "azimuth": f"{pci_col}_azimuth",
            "height": f"{pci_col}_height",
            "pci": pci_col,
        })

        df = df.merge(merge_df, how="left", on=pci_col)
        df[f"{pci_col}_bs_found"] = ~df[f"{pci_col}_lat"].isna()

    return df


def load_downlink_series_data():
    paths = require_files(["v1_5G_DL.xlsx", "v2_5G_DL.xlsx"])
    df = pd.concat([
        pd.read_excel(path, sheet_name="Series Formatted Data") for path in paths
    ], ignore_index=True)

    df = df.loc[:, ~df.columns.duplicated()]
    df = df[["Latitude", "Longitude"] + [c for c in DL_FEATURES if c in df.columns]]
    df = df.dropna(subset=["Latitude", "Longitude"])

    df = df.dropna(subset=SIGNAL_COLS, how="all")
    df["bad_signal"] = df.apply(is_useless_signal, axis=1)
    df = df[~df["bad_signal"]].copy()
    df = df.drop(columns=["bad_signal"])

    df["loc_key"] = df["Latitude"].round(5).astype(str) + "_" + df["Longitude"].round(5).astype(str)

    df = enrich_with_base_stations(df)

    for pci_base in PCI_BASES:
        if f"{pci_base}_lat" in df.columns:
            df = apply_geometry_features(df, pci_base)

    df.to_csv(OUTPUT_DIR / "clean_dl.csv", index=False)
    return df.reset_index(drop=True)


def preview():
    print("Loading and cleaning DL dataset...")
    dl = load_downlink_series_data()
    print("Cleaned DL shape:", dl.shape)


if __name__ == "__main__":
    preview()
