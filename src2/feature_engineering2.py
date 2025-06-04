import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def extract_features_and_labels(df: pd.DataFrame, drop_threshold=0.5, scale=True):
    """
    Extracts cleaned and relevant features + GPS labels (lon, lat).
    Includes signal stats and engineered neighbor features.
    - Fills remaining NaNs with median
    - Scales features with StandardScaler
    """
    y = df[["Longitude", "Latitude"]].values

    # Keep only numeric data
    X = df.select_dtypes(include=[np.number]).drop(columns=["Longitude", "Latitude"])

    # Fill NaNs with median
    X = X.fillna(X.median(numeric_only=True))

    if X.empty:
        print("⚠️ No valid rows after filtering. Returning empty arrays.")
        return np.empty((0, len(X.columns))), np.empty((0, 2)), None

    if scale:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled, y, scaler
    else:
        return X.values, y, None


def preview():
    print("\n🔍 Previewing engineered features...")
    
if __name__ == "__main__":
    preview()
