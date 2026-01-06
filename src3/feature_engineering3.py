import numpy as np
import pandas as pd
from typing import Optional
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from dataloader3 import load_downlink_series_data


def build_dataset() -> pd.DataFrame:
    return load_downlink_series_data()


def select_feature_columns(X: pd.DataFrame, max_features: Optional[int] = 200, variance_threshold: float = 1e-6):
    missing_frac = X.isna().mean()
    X = X.loc[:, missing_frac <= 0.5]

    variances = X.var(skipna=True)
    X = X.loc[:, variances > variance_threshold]

    if max_features is not None and X.shape[1] > max_features:
        top_cols = variances.sort_values(ascending=False).head(max_features).index
        X = X[top_cols]

    return X


def extract_features_and_labels(
    df: pd.DataFrame,
    scale: bool = True,
    scaler: Optional[StandardScaler] = None,
    feature_cols: Optional[list[str]] = None,
    max_features: Optional[int] = 200,
):
    y = df[["Longitude", "Latitude"]].values
    X = df.drop(columns=["Longitude", "Latitude", "loc_key"], errors="ignore")
    X = X.select_dtypes(include=[np.number])
    X = X.replace([np.inf, -np.inf], np.nan)

    if feature_cols is not None:
        X = X.reindex(columns=feature_cols)
    else:
        X = select_feature_columns(X, max_features=max_features)
        feature_cols = list(X.columns)

    X = X.dropna(axis=1, how="all")
    X = X.fillna(X.median(numeric_only=True))

    if scale:
        if scaler is None:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        else:
            X_scaled = scaler.transform(X)
        return X_scaled, y, scaler, feature_cols

    return X.values, y, None, feature_cols


def spatial_kmeans_split(
    df: pd.DataFrame,
    n_clusters: int = 20,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    random_state: int = 42,
):
    coords = df[["Latitude", "Longitude"]].values
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(coords)

    rng = np.random.default_rng(random_state)
    train_idx = []
    val_idx = []
    test_idx = []

    for cluster in np.unique(cluster_labels):
        indices = np.where(cluster_labels == cluster)[0]
        rng.shuffle(indices)
        n_total = len(indices)
        n_test = max(1, int(n_total * test_frac))
        n_val = max(1, int(n_total * val_frac))

        test_idx.extend(indices[:n_test])
        val_idx.extend(indices[n_test:n_test + n_val])
        train_idx.extend(indices[n_test + n_val:])

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    return train_df, val_df, test_df


def preview():
    print("Previewing engineered features for DL dataset")
    df = build_dataset()

    X, y, _, _ = extract_features_and_labels(df)

    print(f"Feature shape: {X.shape}")
    print(f"Label shape: {y.shape}")


if __name__ == "__main__":
    preview()
