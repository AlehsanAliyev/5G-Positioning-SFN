# import pandas as pd
# import os
#
# RAW_DATA_DIR = "data/raw"
#
# def load_excel_stats(filepath: str, stat_filter="Mean") -> pd.DataFrame:
#     """
#     Loads a 5G Excel file and optionally filters rows by the Statistic column.
#     If stat_filter is None, all rows are returned.
#     """
#     df = pd.read_excel(filepath)
#
#     if "Statistic" in df.columns and stat_filter:
#         df = df[df["Statistic"] == stat_filter].reset_index(drop=True)
#
#     return df
#
# def load_downlink_data(stat_filter="Mean") -> pd.DataFrame:
#     path = os.path.join(RAW_DATA_DIR, "5G_DL.xlsx")
#     return load_excel_stats(path, stat_filter)
#
# def load_uplink_data(stat_filter="Mean") -> pd.DataFrame:
#     path = os.path.join(RAW_DATA_DIR, "5G_UL.xlsx")
#     return load_excel_stats(path, stat_filter)
#
# def load_scanner_data(stat_filter="Mean") -> pd.DataFrame:
#     path = os.path.join(RAW_DATA_DIR, "5G_Scanner.xlsx")
#     return load_excel_stats(path, stat_filter)
#
# def load_base_station_config() -> pd.DataFrame:
#     path = os.path.join(RAW_DATA_DIR, "İTÜ 5G Hücre Bilgileri.xlsx")
#     df = pd.read_excel(path)
#
#     # Try to map useful columns like PCI, Lat, Lon
#     col_map = {}
#     for col in df.columns:
#         lower = col.lower()
#         if "latitude" in lower:
#             col_map[col] = "lat"
#         elif "longitude" in lower:
#             col_map[col] = "lon"
#         elif "azimuth" in lower:
#             col_map[col] = "azimuth"
#         elif "height" in lower:
#             col_map[col] = "height"
#         elif "pci" in lower:
#             col_map[col] = "pci"
#
#     df.rename(columns=col_map, inplace=True)
#     return df
#
# def preview():
#     print("📡 Previewing loaded data...\n")
#     dl = load_downlink_data(stat_filter="Mean")
#     ul = load_uplink_data(stat_filter="Mean")
#     sc = load_scanner_data(stat_filter="Mean")
#     bs = load_base_station_config()
#
#     print("✅ Downlink shape:", dl.shape)
#     print("✅ Uplink shape:", ul.shape)
#     print("✅ Scanner shape:", sc.shape)
#     print("✅ Base stations:", bs.shape)
#     print("\n📌 Base Station Sample:")
#     print(bs.head())
#
# if __name__ == "__main__":
#     preview()
import pandas as pd
import os

RAW_DATA_DIR = "../data/raw"

def load_excel_stats(filepath: str, stat_filter="Mean") -> pd.DataFrame:
    """
    Loads a 5G Excel file and optionally filters rows by the Statistic column.
    If stat_filter is None, all rows are returned.
    """
    df = pd.read_excel(filepath)

    if "Statistic" in df.columns and stat_filter:
        df = df[df["Statistic"] == stat_filter].reset_index(drop=True)
    return df

def load_downlink_data(stat_filter="Mean") -> pd.DataFrame:
    path = os.path.join(RAW_DATA_DIR, "5G_DL.xlsx")
    return load_excel_stats(path, stat_filter)

def load_uplink_data(stat_filter="Mean") -> pd.DataFrame:
    path = os.path.join(RAW_DATA_DIR, "5G_UL.xlsx")
    return load_excel_stats(path, stat_filter)

def load_scanner_data(stat_filter="Mean") -> pd.DataFrame:
    path = os.path.join(RAW_DATA_DIR, "5G_Scanner.xlsx")
    return load_excel_stats(path, stat_filter)

def load_base_station_config() -> pd.DataFrame:
    path = os.path.join(RAW_DATA_DIR, "İTÜ 5G Hücre Bilgileri.xlsx")
    df = pd.read_excel(path)

    # Try to map useful columns like PCI, Lat, Lon
    col_map = {}
    for col in df.columns:
        lower = col.lower()
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

    df.rename(columns=col_map, inplace=True)
    return df

def load_downlink_series_data() -> pd.DataFrame:
    """
    Loads the 'Series Formatted Data' sheet from DL, including real GPS.
    Returns only numeric + positional data.
    """
    path = os.path.join(RAW_DATA_DIR, "5G_DL.xlsx")
    df = pd.read_excel(path, sheet_name="Series Formatted Data")

    # Drop rows without GPS
    df = df[df["Latitude"].notna() & df["Longitude"].notna()]
    df = df.select_dtypes(include=["number"]).copy()
    df.rename(columns={"Latitude": "lat", "Longitude": "lon"}, inplace=True)

    return df

def preview():
    print("📡 Previewing loaded data...\n")
    dl = load_downlink_data(stat_filter="Mean")
    ul = load_uplink_data(stat_filter="Mean")
    sc = load_scanner_data(stat_filter="Mean")
    bs = load_base_station_config()
    gps = load_downlink_series_data()

    print("✅ Downlink shape:", dl.shape)
    print("✅ Uplink shape:", ul.shape)
    print("✅ Scanner shape:", sc.shape)
    print("✅ Base stations:", bs.shape)
    print("✅ Series Data (real GPS):", gps.shape)

    print("\n📌 GPS Sample:")
    print(gps[["lat", "lon"]].head())

if __name__ == "__main__":
    preview()
