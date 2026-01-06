from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dataloader3 import load_downlink_series_data
from feature_engineering3 import extract_features_and_labels


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs" / "diagnostics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def summarize_missingness(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.select_dtypes(include=[np.number])
    missing = numeric.isna().mean().sort_values(ascending=False)
    return missing.reset_index().rename(columns={"index": "column", 0: "missing_frac"})


def summarize_variance(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.select_dtypes(include=[np.number])
    var = numeric.var(skipna=True).sort_values(ascending=False)
    return var.reset_index().rename(columns={"index": "column", 0: "variance"})


def quick_correlation(df: pd.DataFrame, target_cols: list[str]) -> pd.DataFrame:
    numeric = df.select_dtypes(include=[np.number])
    corr = numeric.corr(numeric_only=True)
    rows = []
    for target in target_cols:
        if target in corr.columns:
            series = corr[target].drop(target, errors="ignore").abs().sort_values(ascending=False)
            for col, val in series.head(25).items():
                rows.append({"target": target, "feature": col, "abs_corr": val})
    return pd.DataFrame(rows)


def plot_signal_distributions(df: pd.DataFrame) -> None:
    cols = [c for c in df.columns if any(k in c for k in ["RSRP", "RSRQ", "SINR"])]
    cols = [c for c in cols if df[c].dtype != object][:6]
    if not cols:
        return
    fig, axes = plt.subplots(len(cols), 1, figsize=(8, 2.5 * len(cols)))
    if len(cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, cols):
        ax.hist(df[col].dropna(), bins=50, color="steelblue", edgecolor="black")
        ax.set_title(f"Distribution: {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "signal_distributions.png")
    plt.show()


def plot_geo_spread(df: pd.DataFrame) -> None:
    if "Latitude" not in df.columns or "Longitude" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(df["Longitude"], df["Latitude"], s=5, alpha=0.5)
    ax.set_title("GPS Coverage")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "gps_coverage.png")
    plt.show()


def main():
    print("Loading DL dataset for diagnostics...")
    df = load_downlink_series_data()

    missing = summarize_missingness(df)
    missing.to_csv(OUTPUT_DIR / "missingness.csv", index=False)

    variance = summarize_variance(df)
    variance.to_csv(OUTPUT_DIR / "variance.csv", index=False)

    corr = quick_correlation(df, target_cols=["Latitude", "Longitude"])
    corr.to_csv(OUTPUT_DIR / "top_correlations.csv", index=False)

    plot_signal_distributions(df)
    plot_geo_spread(df)

    X, y, _scaler, feature_cols = extract_features_and_labels(df, scale=True)
    summary = {
        "rows": int(len(df)),
        "feature_cols": int(len(feature_cols)),
        "lat_min": float(df["Latitude"].min()),
        "lat_max": float(df["Latitude"].max()),
        "lon_min": float(df["Longitude"].min()),
        "lon_max": float(df["Longitude"].max()),
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Diagnostics saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
