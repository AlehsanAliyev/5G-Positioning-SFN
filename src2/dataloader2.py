import pandas as pd
import numpy as np
import os

RAW_DATA_DIR = "../5G-Positioning-SFN/data/raw"

dl_ul_important = [
                "NR_UE_RSRP_0",
                "NR_UE_RSRQ_0",
                "NR_UE_SINR_0",
                "NR_UE_Nbr_PCI_0",
                "NR_UE_Nbr_PCI_1",
                "NR_UE_Nbr_PCI_2",
                "NR_UE_Nbr_PCI_3",
                "NR_UE_Nbr_RSRP_0",
                "NR_UE_Nbr_RSRP_1",
                "NR_UE_Nbr_RSRP_2",
                "NR_UE_Nbr_RSRP_3",
                "NR_UE_Nbr_RSRQ_0",
                "NR_UE_Nbr_RSRQ_1",
                "NR_UE_Nbr_RSRQ_2",
                "NR_UE_Nbr_RSRQ_3"]

scan_important = [
                    "NR_Scan_PCI_SortedBy_RSRP_0",
                    "NR_Scan_PCI_SortedBy_RSRP_1",
                    "NR_Scan_PCI_SortedBy_RSRP_2",
                    "NR_Scan_PCI_SortedBy_RSRP_3",
                    "NR_Scan_PCI_SortedBy_RSRP_4",
                    "NR_Scan_PCI_SortedBy_RSRP_5",
                    "NR_Scan_PCI_SortedBy_RSRP_6",
                    "NR_Scan_SSB_RSRP_SortedBy_RSRP_0",
                    "NR_Scan_SSB_RSRP_SortedBy_RSRP_1",
                    "NR_Scan_SSB_RSRP_SortedBy_RSRP_2",
                    "NR_Scan_SSB_RSRP_SortedBy_RSRP_3",
                    "NR_Scan_SSB_RSRP_SortedBy_RSRP_4",
                    "NR_Scan_SSB_RSRP_SortedBy_RSRP_5",
                    "NR_Scan_SSB_RSRP_SortedBy_RSRP_6",
                    "NR_Scan_SSB_RSRQ_SortedBy_RSRP_0",
                    "NR_Scan_SSB_RSRQ_SortedBy_RSRP_1",
                    "NR_Scan_SSB_RSRQ_SortedBy_RSRP_2",
                    "NR_Scan_SSB_RSRQ_SortedBy_RSRP_3",
                    "NR_Scan_SSB_RSRQ_SortedBy_RSRP_4",
                    "NR_Scan_SSB_RSRQ_SortedBy_RSRP_5",
                    "NR_Scan_SSB_RSRQ_SortedBy_RSRP_6",
                    "NR_Scan_SSB_SINR_SortedBy_RSRP_0",
                    "NR_Scan_SSB_SINR_SortedBy_RSRP_1",
                    "NR_Scan_SSB_SINR_SortedBy_RSRP_2",
                    "NR_Scan_SSB_SINR_SortedBy_RSRP_3",
                    "NR_Scan_SSB_SINR_SortedBy_RSRP_4",
                    "NR_Scan_SSB_SINR_SortedBy_RSRP_5",
                    "NR_Scan_SSB_SINR_SortedBy_RSRP_6"]

def clean_df(df, important_columns):
    df = df.copy()
    df = df.select_dtypes(include=[np.number])  # numeric only
    df = df.dropna(subset=important_columns, how='all')
    df = df.dropna(subset=["Latitude", "Longitude"] )
    # df = df.dropna()  # drop rows with NaNs (optional for now)
    
    # Drop all other columns
    needed_columns = ["Latitude", "Longitude", "NR_UE_PCI_0"] + important_columns
    df = df[[col for col in needed_columns if col in df.columns]]
    
    return df

def load_downlink_series_data() -> pd.DataFrame:
    """
    Loads the 'Series Formatted Data' sheet from the downlink Excel file,
    ensuring it includes GPS labels and numeric signal data.
    """
    path = os.path.join(RAW_DATA_DIR, "v1_5G_DL.xlsx")
    df = pd.read_excel(path, sheet_name="Series Formatted Data")
    
    path2 = os.path.join(RAW_DATA_DIR, "v2_5G_DL.xlsx")
    df2 = pd.read_excel(path2, sheet_name="Series Formatted Data")
    df = pd.concat([df, df2], ignore_index=True)

    df = clean_df(df, dl_ul_important)
    return df.reset_index(drop=True)

def load_uplink_series_data() -> pd.DataFrame:
    path = os.path.join(RAW_DATA_DIR, "v1_5G_UL.xlsx")
    df = pd.read_excel(path, sheet_name="Series Formatted Data")
    
    path2 = os.path.join(RAW_DATA_DIR, "v2_5G_UL.xlsx")
    df2 = pd.read_excel(path2, sheet_name="Series Formatted Data")
    df = pd.concat([df, df2], ignore_index=True)

    
    df = clean_df(df, dl_ul_important)
    return df

def load_scanner_series_data() -> pd.DataFrame:
    path = os.path.join(RAW_DATA_DIR, "v1_5G_Scanner.xlsx")
    df = pd.read_excel(path, sheet_name="Series Formatted Data")

    
    path2 = os.path.join(RAW_DATA_DIR, "v2_5G_Scanner.xlsx")
    df2 = pd.read_excel(path2, sheet_name="Series Formatted Data")
    df = pd.concat([df, df2], ignore_index=True)

    df = clean_df(df, scan_important)
    return df

def load_base_station_config() -> pd.DataFrame:
    path = os.path.join(RAW_DATA_DIR, "İTÜ 5G Hücre Bilgileri.xlsx")
    df = pd.read_excel(path, sheet_name="Hücre tablosu")

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
    needed_columns = ["lat", "lon", "azimuth", "height", "pci"]
    df = df[[col for col in needed_columns if col in df.columns]]
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