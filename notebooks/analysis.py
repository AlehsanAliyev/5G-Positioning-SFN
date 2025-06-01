import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Setup ---
os.makedirs("analysis_outputs", exist_ok=True)

# --- Load Data ---
dl1 = pd.read_excel("../5G-Positioning-SFN/data/raw/v1_5G_DL.xlsx", sheet_name="Series Formatted Data")
ul1 = pd.read_excel("../5G-Positioning-SFN/data/raw/v1_5G_UL.xlsx", sheet_name="Series Formatted Data")
scanner1 = pd.read_excel("../5G-Positioning-SFN/data/raw/v1_5G_Scanner.xlsx", sheet_name="Series Formatted Data")

dl2 = pd.read_excel("../5G-Positioning-SFN/data/raw/v2_5G_DL.xlsx", sheet_name="Series Formatted Data")
ul2 = pd.read_excel("../5G-Positioning-SFN/data/raw/v2_5G_UL.xlsx", sheet_name="Series Formatted Data")
scanner2 = pd.read_excel("../5G-Positioning-SFN/data/raw/v2_5G_Scanner.xlsx", sheet_name="Series Formatted Data")

dl = pd.concat([dl1, dl2], ignore_index=True)
ul = pd.concat([ul1, ul2], ignore_index=True)
scanner = pd.concat([scanner1, scanner2], ignore_index=True)

# --- Clean & Filter ---
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

dl_clean = clean_df(dl, dl_ul_important)
dl_clean.to_csv("analysis_outputs/DL_clean.csv", index=False)

ul_clean = clean_df(ul, dl_ul_important)
ul_clean.to_csv("analysis_outputs/UL_clean.csv", index=False)

scanner_clean = clean_df(scanner, scan_important)
scanner_clean.to_csv("analysis_outputs/scanner_clean.csv", index=False)

# --- Save Missing Value Summary ---
missing_summary = pd.DataFrame({
    'DL_Missing%': dl.isnull().mean() * 100,
    'UL_Missing%': ul.isnull().mean() * 100,
    'Scanner_Missing%': scanner.isnull().mean() * 100
})
missing_summary.to_csv("analysis_outputs/missing_value_summary.csv")

# --- Variance Summary ---
dl_var = dl_clean.var().sort_values(ascending=False).head(20)
ul_var = ul_clean.var().sort_values(ascending=False).head(20)
scanner_var = scanner_clean.var().sort_values(ascending=False).head(20)

dl_var.to_csv("analysis_outputs/top_variance_DL.csv")
ul_var.to_csv("analysis_outputs/top_variance_UL.csv")
scanner_var.to_csv("analysis_outputs/top_variance_Scanner.csv")

# --- Correlation Heatmap ---
def plot_corr_heatmap(df, name):
    plt.figure(figsize=(12, 10))
    corr = df.corr()
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title(f"{name} - Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(f"analysis_outputs/corr_{name}.png")
    plt.close()

plot_corr_heatmap(dl_clean, "DL")
plot_corr_heatmap(ul_clean, "UL")
plot_corr_heatmap(scanner_clean, "Scanner")

print("✅ Analysis done! Check 'analysis_outputs/' folder for results.")
