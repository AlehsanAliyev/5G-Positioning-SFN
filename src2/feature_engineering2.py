import numpy as np
import pandas as pd
from typing import Optional
from sklearn.preprocessing import StandardScaler

from dataloader2 import load_downlink_series_data


def build_dataset() -> pd.DataFrame:
    return load_downlink_series_data()


def extract_features_and_labels(
    df: pd.DataFrame,
    scale: bool = True,
    scaler: Optional[StandardScaler] = None,
):
    y = df[["Longitude", "Latitude"]].values
    X = df.drop(columns=["Longitude", "Latitude", "loc_key"], errors="ignore")
    X = X.select_dtypes(include=[np.number])
    X = X.replace([np.inf, -np.inf], np.nan)
    # Drop columns that are entirely NaN after cleaning.
    X = X.dropna(axis=1, how="all")
    X = X.fillna(X.median(numeric_only=True))

    if scale:
        if scaler is None:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        else:
            X_scaled = scaler.transform(X)
        return X_scaled, y, scaler

    return X.values, y, None


def preview():
    print("Previewing engineered features for DL dataset")
    df = build_dataset()

    X, y, _ = extract_features_and_labels(df)

    print(f"Feature shape: {X.shape}")
    print(f"Label shape: {y.shape}")


if __name__ == "__main__":
    preview()
