from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# Ensure output directory
os.makedirs("analysis_outputs_additional", exist_ok=True)


def advanced_analysis(df_raw, y, name):
    df = df_raw.copy()

    # Drop label columns if present
    df = df.drop(columns=[col for col in df.columns if col.lower() in ["latitude", "longitude", "lat", "lon"]],
                 errors='ignore')

    # Drop non-numeric and high-missing columns
    df = df.select_dtypes(include=[np.number])
    df = df.dropna(axis=1, thresh=len(df) * 0.5).dropna()

    if df.shape[1] < 3 or len(df) != len(y):
        print(f"⚠️ Skipping {name} due to insufficient usable data")
        return

    # Align y to df's index
    y = y[:len(df)]

    # Standardize
    X_scaled = StandardScaler().fit_transform(df)

    # --- 1. PCA ---
    pca = PCA()
    pca.fit(X_scaled)
    explained = np.cumsum(pca.explained_variance_ratio_) * 100

    plt.figure()
    plt.plot(explained, marker='o')
    plt.title(f"{name} PCA Explained Variance")
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Variance Explained (%)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"analysis_outputs/pca_{name}.png")
    plt.close()

    # --- 2. Mutual Info (on Longitude as target example) ---
    mi = mutual_info_regression(X_scaled, y[:, 0])
    pd.DataFrame({
        "Feature": df.columns,
        "MI_with_Longitude": mi
    }).sort_values(by="MI_with_Longitude", ascending=False) \
        .to_csv(f"analysis_outputs/mi_{name}.csv", index=False)

    # --- 3. LightGBM Feature Importance ---
    model = lgb.LGBMRegressor(n_estimators=100)
    model.fit(df, y[:, 0])
    fi = pd.Series(model.feature_importances_, index=df.columns)
    fi.sort_values(ascending=False).to_csv(f"analysis_outputs/importance_{name}.csv")

    print(f"✅ Analysis completed for {name}.")


# --- Load GPS Labels for DL (used across all for consistency) ---
dl = pd.read_excel("../data/raw/5G_DL.xlsx", sheet_name="Series Formatted Data")
gps_labels = dl[["Longitude", "Latitude"]].dropna().reset_index(drop=True).values

# --- Load All Datasets ---
datasets = {
    "DL": pd.read_excel("../data/raw/5G_DL.xlsx", sheet_name="Series Formatted Data"),
    "UL": pd.read_excel("../data/raw/5G_UL.xlsx", sheet_name="Series Formatted Data"),
    "Scanner": pd.read_excel("../data/raw/5G_Scanner.xlsx", sheet_name="Series Formatted Data")
}

for name, df in datasets.items():
    print(f"\n🔍 Processing {name}...")

    # Extract labels from *that* dataset (not DL only)
    if "Longitude" in df.columns and "Latitude" in df.columns:
        df = df.dropna(subset=["Longitude", "Latitude"])
        gps = df[["Longitude", "Latitude"]].values
    else:
        print(f"⚠️ Skipping {name}: No GPS coordinates in dataset")
        continue

    print(f"🧹 Raw shape: {df.shape}")

    # Run analysis
    advanced_analysis(df, gps, name)
