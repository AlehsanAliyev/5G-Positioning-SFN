import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from feature_engineering import merge_signal_data, extract_features

def cluster_and_plot(n_clusters=4):
    # Load merged signal data
    merged = merge_signal_data(stat_filter=None)
    X = extract_features(merged)

    # Step 1: Keep only numeric columns
    X_numeric = X.select_dtypes(include=["number"])

    # Step 2: Drop columns that are fully NaN
    X_numeric = X_numeric.dropna(axis=1, how="all")

    # Step 3: Fill any remaining NaNs with median
    X_filled = X_numeric.fillna(X_numeric.median())

    # Step 4: Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_filled)

    # Step 5: PCA to 2D
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Step 6: KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X_scaled)

    # Step 7: Plot
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="viridis", s=70, edgecolors='k')
    plt.title("📊 PCA + KMeans Clustering of 5G Signal Stats", fontsize=14)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.colorbar(scatter, label="Cluster Label")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    cluster_and_plot(n_clusters=4)
