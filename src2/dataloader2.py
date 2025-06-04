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
                "NR_UE_Nbr_RSRQ_3",
                "NR_UE_Throughput_PDCP_DL",
                "NR_UE_RB_Num_DL_0",
                "NR_UE_Pathloss_DL_0",
                "NR_UE_Power_Tx_PUSCH_0"]

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


def merge_signal_data(df) -> pd.DataFrame:
    df = df.dropna(subset=["Latitude", "Longitude"])
    if "NR_UE_PCI_0" in df.columns:
        df = df.dropna(subset=["NR_UE_PCI_0"])

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
    valid = series[(series > threshold) & (~series.isna())]
    return valid.mean() if not valid.empty else np.nan


# Modified function to return two separate aligned DataFrames for UL and DL models

def prepare(df, prefix = ""):
    df = df.copy()
    df['loc_key'] = df["Latitude"].round(5).astype(str) + "_" + df["Longitude"].round(5).astype(str)
    signal_cols = [col for col in df.columns if col not in ["loc_key"]]
    
    agg_funcs = {
        col: (lambda x, col=col: masked_mean(x, get_threshold(col)))
        for col in signal_cols
    }
    grouped = df.groupby("loc_key").agg(agg_funcs)
    grouped.columns = [f"{prefix}{col}" for col in grouped.columns]
    grouped = merge_signal_data(grouped)
    return grouped

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

    df = clean_df(df, dl_ul_important + ["App_Throughput_DL"])
    df = prepare(df)
    return df.reset_index(drop=True)

def load_uplink_series_data() -> pd.DataFrame:
    path = os.path.join(RAW_DATA_DIR, "v1_5G_UL.xlsx")
    df = pd.read_excel(path, sheet_name="Series Formatted Data")
    
    path2 = os.path.join(RAW_DATA_DIR, "v2_5G_UL.xlsx")
    df2 = pd.read_excel(path2, sheet_name="Series Formatted Data")
    df = pd.concat([df, df2], ignore_index=True)

    
    df = clean_df(df, dl_ul_important + ["App_Throughput_UL"])
    df = prepare(df)
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
    dl.to_csv("analysis_outputs/clean_dl.csv", index=False)
    ul = load_uplink_series_data()
    ul.to_csv("analysis_outputs/clean_ul.csv", index=False)
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