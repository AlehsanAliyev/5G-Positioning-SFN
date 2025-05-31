import pandas as pd
import os

RAW_DATA_DIR = "../data/raw"

def load_downlink_series_data() -> pd.DataFrame:
    """
    Loads the 'Series Formatted Data' sheet from the downlink Excel file,
    ensuring it includes GPS labels and numeric signal data.
    """
    path = os.path.join(RAW_DATA_DIR, "5G_DL.xlsx")
    df = pd.read_excel(path, sheet_name="Series Formatted Data")

    # Keep only rows with valid GPS data
    df = df[df["Latitude"].notna() & df["Longitude"].notna()]
    return df.reset_index(drop=True)

def load_uplink_series_data() -> pd.DataFrame:
    path = os.path.join(RAW_DATA_DIR, "5G_UL.xlsx")
    df = pd.read_excel(path, sheet_name="Series Formatted Data")
    return df

def load_scanner_series_data() -> pd.DataFrame:
    path = os.path.join(RAW_DATA_DIR, "5G_Scanner.xlsx")
    df = pd.read_excel(path, sheet_name="Series Formatted Data")
    return df

def load_base_station_config() -> pd.DataFrame:
    path = os.path.join(RAW_DATA_DIR, "İTÜ 5G Hücre Bilgileri.xlsx")
    df = pd.read_excel(path)

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

def preview():
    print("\U0001F4F0 Previewing loaded data...\n")
    dl = load_downlink_series_data()
    ul = load_uplink_series_data()
    sc = load_scanner_series_data()
    bs = load_base_station_config()

    print("✅ Downlink Series shape:", dl.shape)
    print("✅ Uplink Series shape:", ul.shape)
    print("✅ Scanner Series shape:", sc.shape)
    print("✅ Base Station Config shape:", bs.shape)
    print("\n📌 GPS Sample from Downlink:")
    print(dl[["Latitude", "Longitude"]].head())

if __name__ == "__main__":
    preview()