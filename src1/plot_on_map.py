import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point
from feature_engineering import merge_signal_data, extract_features
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def simulate_coordinates(n, lat_range=(41.104, 41.107), lon_range=(29.025, 29.030)):
    """Generate n fake GPS points within ITÜ Ayazağa Campus bounds."""
    lats = np.linspace(lat_range[0], lat_range[1], n)
    lons = np.linspace(lon_range[0], lon_range[1], n)
    return pd.DataFrame({"Latitude": lats, "Longitude": lons})

def cluster_signals(n_clusters=4):
    merged = merge_signal_data(stat_filter=None)
    X = extract_features(merged).select_dtypes(include=["number"]).dropna(axis=1, how="all")
    X = X.fillna(X.median())
    X_scaled = StandardScaler().fit_transform(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    labels = KMeans(n_clusters=n_clusters, random_state=42).fit_predict(X_scaled)
    return labels, len(X)

def plot_on_gis_map():
    # ✅ Windows-compatible local path to shapefile
    shapefile_path = r"data\raw\maps\ITU_3DBINA_EPSG4326.shp"
    buildings = gpd.read_file(shapefile_path)

    # Generate clusters & coordinates
    labels, n = cluster_signals()
    coords_df = simulate_coordinates(n)

    # Create clustered GeoDataFrame
    geometry = [Point(lon, lat) for lat, lon in zip(coords_df["Latitude"], coords_df["Longitude"])]
    gdf_points = gpd.GeoDataFrame(coords_df, geometry=geometry, crs="EPSG:4326")
    gdf_points["cluster"] = labels

    # Plot it all
    fig, ax = plt.subplots(figsize=(10, 10))
    buildings.plot(ax=ax, color="lightgray", edgecolor="black")
    gdf_points.plot(ax=ax, column="cluster", cmap="viridis", markersize=60, legend=True)

    plt.title("📍 5G Signal Clusters Over ITÜ Campus")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_on_gis_map()
