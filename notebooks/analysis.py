import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Setup ---
os.makedirs("analysis_outputs", exist_ok=True)

# --- Load Data ---
dl = pd.read_excel("../data/raw/5G_DL.xlsx", sheet_name="Series Formatted Data")
ul = pd.read_excel("../data/raw/5G_UL.xlsx", sheet_name="Series Formatted Data")
scanner = pd.read_excel("../data/raw/5G_Scanner.xlsx", sheet_name="Series Formatted Data")

# --- Clean & Filter ---
def clean_df(df):
    df = df.copy()
    df = df.select_dtypes(include=[np.number])  # numeric only
    df = df.dropna(axis=1, thresh=len(df)*0.5)  # drop columns >50% NaN
    df = df.dropna()  # drop rows with NaNs (optional for now)
    return df

dl_clean = clean_df(dl)
ul_clean = clean_df(ul)
scanner_clean = clean_df(scanner)

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
