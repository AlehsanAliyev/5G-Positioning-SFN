import os
import pandas as pd
import geopandas as gpd
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import gc
from pyproj import CRS
from rasterio.plot import show
from rasterio.windows import Window

# Directory where maps are already unzipped
EXTRACT_DIR = '../data/raw/maps'
# Directory for saving visual outputs instead of shapefiles
VIS_DIR = 'visualizations2'
os.makedirs(VIS_DIR, exist_ok=True)

# 1. Metadata summary for all layers (vector & raster)
import fiona

def parse_metadata(extract_dir=EXTRACT_DIR):
    """
    Summarizes metadata for each shapefile and raster in the extract directory.
    For shapefiles: driver, CRS, feature count, geometry type.
    For ASCII grids: driver, dtype, nodata, dimensions, CRS.
    """
    records = []
    # Vector layers
    for root, _, files in os.walk(extract_dir):
        for fname in files:
            path = os.path.join(root, fname)
            key = fname.lower()
            if key.endswith('.shp'):
                layer = os.path.splitext(fname)[0]
                with fiona.open(path) as src:
                    rec = {
                        'LAYER_NAME': layer,
                        'TYPE': 'vector',
                        'driver': src.driver,
                        'crs': src.crs_wkt,
                        'num_features': len(src),
                        'geometry': src.schema['geometry']
                    }
                records.append(rec)
            elif key.endswith('.asc'):
                layer = os.path.splitext(fname)[0]
                with rasterio.open(path) as src:
                    rec = {
                        'LAYER_NAME': layer,
                        'TYPE': 'raster',
                        'driver': src.driver,
                        'dtype': src.dtypes[0],
                        'nodata': src.nodata,
                        'width': src.width,
                        'height': src.height,
                        'crs': src.crs.to_string()
                    }
                records.append(rec)
    df = pd.DataFrame(records)
    # Order columns
    cols = ['LAYER_NAME', 'TYPE'] + [c for c in df.columns if c not in ('LAYER_NAME','TYPE')]
    return df[cols]

# 2. Catalog attribute definitions for each shapefile Catalog attribute definitions for each shapefile
def catalog_attributes(extract_dir=EXTRACT_DIR):
    records = []
    for root, _, files in os.walk(extract_dir):
        for fname in files:
            if fname.lower().endswith('.shp'):
                path = os.path.join(root, fname)
                layer = os.path.splitext(fname)[0]
                gdf = gpd.read_file(path)
                for field, ftype in gdf.dtypes.items():
                    records.append({'LAYER': layer, 'FIELD_NAME': field, 'FIELD_TYPE': str(ftype)})
                del gdf
    return pd.DataFrame(records)

# 3. Check for placeholders ('...') in metadata
def check_placeholders(df):
    mask = df.apply(lambda row: row.astype(str).str.contains(r'\.\.\.', na=False), axis=1)
    return df[mask]

# 4. Summarize CRS directly from shapefiles (no .prj needed)
def summarize_crs(extract_dir=EXTRACT_DIR):
    """
    Reads the CRS directly from each shapefile using GeoPandas and returns
    a DataFrame with LAYER, EPSG, and CRS_WKT. Does not require .prj files.
    """
    records = []
    for root, _, files in os.walk(extract_dir):
        for fname in files:
            if fname.lower().endswith('.shp'):
                layer = os.path.splitext(fname)[0]
                shp_path = os.path.join(root, fname)
                gdf = gpd.read_file(shp_path)
                # use shapefile's CRS (fallback to EPSG:4326 if None)
                crs = gdf.crs if gdf.crs is not None else CRS.from_epsg(4326)
                epsg = crs.to_epsg()
                wkt = crs.to_wkt()
                records.append({'LAYER': layer, 'EPSG': epsg, 'CRS_WKT': wkt})
                del gdf
    df = pd.DataFrame(records)
    return df

# 5. Inspect shapefiles (prints only) Inspect shapefiles (prints only) Inspect shapefiles (prints only)
def inspect_shapefiles(extract_dir=EXTRACT_DIR, n=5):
    for root, _, files in os.walk(extract_dir):
        for fname in files:
            if fname.lower().endswith('.shp'):
                path = os.path.join(root, fname)
                layer = os.path.splitext(fname)[0]
                gdf = gpd.read_file(path)
                print(f"Layer: {layer} (CRS: {gdf.crs})")
                print(gdf.head(n))
                print('-' * 40)
                del gdf
                gc.collect()

# 6. Analyze raster DEM and plot histogram (float32)
def analyze_raster(extract_dir=EXTRACT_DIR):
    for root, _, files in os.walk(extract_dir):
        for fname in files:
            if fname.lower().endswith('.asc'):
                path = os.path.join(root, fname)
                with rasterio.open(path) as src:
                    arr = src.read(1, out_dtype='float32')
                    nodata = src.nodata
                    valid = arr[arr != nodata]
                    print(f"Raster: {fname} — Min: {valid.min()}, Max: {valid.max()}, Mean: {valid.mean():.2f}")
                    fig, ax = plt.subplots()
                    ax.hist(valid.flatten(), bins=50)
                    ax.set(title=f"Elevation Distribution: {fname}", xlabel='Elevation', ylabel='Frequency')
                    out_hist = os.path.join(VIS_DIR, f"hist_{os.path.splitext(fname)[0]}.png")
                    fig.savefig(out_hist)
                    plt.close(fig)
                    print(f"Saved histogram to {out_hist}")
                del arr, valid
                gc.collect()
                return

# 7. Generate slope and aspect rasters in windows (no change)
def generate_slope_aspect(extract_dir=EXTRACT_DIR, slope_file='slope.tif', aspect_file='aspect.tif'):
    for root, _, files in os.walk(extract_dir):
        for fname in files:
            if fname.lower().endswith('.asc'):
                path = os.path.join(root, fname)
                with rasterio.open(path) as src:
                    meta = src.meta.copy()
                    meta.update(dtype='float32', count=1)
                    with rasterio.open(slope_file, 'w', **meta) as ds_s, rasterio.open(aspect_file, 'w', **meta) as ds_a:
                        for _, window in src.block_windows(1):
                            if window.width < 2 or window.height < 2:
                                continue
                            data = src.read(1, window=window, out_dtype='float32')
                            data[data == src.nodata] = np.nan
                            try:
                                dzdy, dzdx = np.gradient(data, src.res[1], src.res[0])
                            except ValueError:
                                continue
                            slope = np.degrees(np.arctan(np.sqrt(dzdx**2 + dzdy**2))).astype('float32')
                            aspect = (np.degrees(np.arctan2(dzdy, -dzdx)) % 360).astype('float32')
                            ds_s.write(np.nan_to_num(slope, nan=src.nodata), 1, window=window)
                            ds_a.write(np.nan_to_num(aspect, nan=src.nodata), 1, window=window)
                    print(f"Slope saved to {slope_file}, aspect saved to {aspect_file}")
                gc.collect()
                return

# 8. Attribute summary (statistics & value counts)
def summarize_attributes(extract_dir=EXTRACT_DIR, out_dir=os.path.join(VIS_DIR,'attr_stats')):
    os.makedirs(out_dir, exist_ok=True)
    for root, _, files in os.walk(extract_dir):
        for fname in files:
            if fname.lower().endswith('.shp'):
                path = os.path.join(root, fname)
                layer = os.path.splitext(fname)[0]
                gdf = gpd.read_file(path)
                stats = gdf.describe().T
                stats.to_csv(os.path.join(out_dir, f"{layer}_describe.csv"))
                del gdf
                gc.collect()
                print(f"Saved attribute stats for {layer}")

# 9. Visualize all vector layers overlay
def visualize_vectors(extract_dir=EXTRACT_DIR, out_file=os.path.join(VIS_DIR,'vectors_overlay.png')):
    fig, ax = plt.subplots()
    for root, _, files in os.walk(extract_dir):
        for fname in files:
            if fname.lower().endswith('.shp'):
                gdf = gpd.read_file(os.path.join(root, fname)).to_crs(epsg=4326)
                gdf.plot(ax=ax, alpha=0.5)
                del gdf
                gc.collect()
    ax.set_title('Overlay of All Vector Layers')
    fig.savefig(out_file)
    plt.close(fig)
    print(f"Saved overlay to {out_file}")

# 10. Visualize DEM
def visualize_dem(extract_dir=EXTRACT_DIR, out_file=os.path.join(VIS_DIR,'dem_plot.png')):
    for root, _, files in os.walk(extract_dir):
        for fname in files:
            if fname.lower().endswith('.asc'):
                with rasterio.open(os.path.join(root, fname)) as src:
                    fig, ax = plt.subplots()
                    show(src, ax=ax)
                    ax.set_title(f"DEM: {fname}")
                    fig.savefig(out_file)
                    plt.close(fig)
                    print(f"Saved DEM plot to {out_file}")
                return

# 11. Thematic maps for each layer (unchanged)
def thematic_maps(extract_dir=EXTRACT_DIR, out_dir=os.path.join(VIS_DIR,'maps_thematic')):
    os.makedirs(out_dir, exist_ok=True)
    for root, _, files in os.walk(extract_dir):
        for fname in files:
            if fname.lower().endswith('.shp'):
                layer = os.path.splitext(fname)[0]
                gdf = gpd.read_file(os.path.join(root, fname)).to_crs(epsg=4326)
                nums = gdf.select_dtypes(include=[np.number]).columns
                if len(nums):
                    fig, ax = plt.subplots()
                    gdf.plot(column=nums[0], legend=True, ax=ax)
                    ax.set_title(f"{layer} by {nums[0]}")
                    fig.savefig(os.path.join(out_dir, f"{layer}_{nums[0]}.png"))
                    plt.close(fig)
                del gdf
                gc.collect()
                print(f"Map for {layer} done")

# 12. Compute building volumes -> visualize only
def compute_building_volume(extract_dir=EXTRACT_DIR):
    for root, _, files in os.walk(extract_dir):
        for fname in files:
            if '3dbina' in fname.lower() and fname.lower().endswith('.shp'):
                path = os.path.join(root, fname)
                gdf = gpd.read_file(path)
                h = next((c for c in gdf.columns if 'YUKSEK' in c.upper() or 'HEIGHT' in c.upper()), None)
                if h:
                    gdf = gdf.to_crs(epsg=3857)
                    gdf['AREA'] = gdf.geometry.area
                    gdf['VOLUME'] = gdf['AREA'] * gdf[h]
                    gdf = gdf.to_crs(epsg=4326)
                    fig, ax = plt.subplots()
                    gdf.plot(column='VOLUME', legend=True, ax=ax)
                    ax.set_title('Building Volume')
                    out_img = os.path.join(VIS_DIR, 'buildings_volume.png')
                    fig.savefig(out_img)
                    plt.close(fig)
                    print(f"Saved building volumes plot to {out_img}")
                del gdf
                gc.collect()
                return

# 13. Buildings within buffer of roads -> visualize only
def buildings_near_roads(extract_dir=EXTRACT_DIR, buffer_dist=50):
    roads, buildings = None, None
    for root, _, files in os.walk(extract_dir):
        for fname in files:
            low = fname.lower()
            if 'ulasimagi' in low and low.endswith('.shp'):
                roads = gpd.read_file(os.path.join(root, fname)).to_crs(epsg=3857)
            if '3dbina' in low and low.endswith('.shp'):
                buildings = gpd.read_file(os.path.join(root, fname)).to_crs(epsg=3857)
        if roads is not None and buildings is not None:
            break
    if roads is None or buildings is None:
        print("Required layers for spatial analysis not found.")
        return
    # Clean geometries
    roads['geometry'] = roads.geometry.buffer(0)
    buildings['geometry'] = buildings.geometry.buffer(0)
    # Create and union buffer geometry
    buffered = roads.geometry.buffer(buffer_dist)
    unioned = buffered.unary_union
    # Check union validity
    if unioned is None or (hasattr(unioned, 'is_empty') and unioned.is_empty):
        print(f"No buffer geometry generated for roads; skipping building proximity.")
        return
    # Filter valid buildings
    valid_buildings = buildings[buildings.is_valid]
    try:
        mask = valid_buildings.geometry.intersects(unioned)
    except Exception as e:
        print(f"Intersection error: {e}")
        return
    nearby = valid_buildings[mask]
    if nearby.empty:
        print(f"No buildings found within {buffer_dist}m of roads.")
        return
    # Reproject for visualization
    roads_wgs = roads.to_crs(epsg=4326)
    nearby_wgs = nearby.to_crs(epsg=4326)
    # Plot
    fig, ax = plt.subplots()
    roads_wgs.plot(ax=ax, color='black', linewidth=1, label='Roads')
    nearby_wgs.plot(ax=ax, color='red', label='Nearby Buildings')
    ax.legend()
    ax.set_title(f"Buildings within {buffer_dist}m of Roads")
    out_img = os.path.join(VIS_DIR, 'buildings_near_roads.png')
    fig.savefig(out_img)
    plt.close(fig)
    print(f"Saved nearby buildings plot to {out_img}")
    # Cleanup
    del roads, buildings, valid_buildings, nearby, buffered, unioned, roads_wgs, nearby_wgs
    gc.collect()

# Main execution
def main():
    # Summaries
    parse_metadata().to_csv('metadata_summary.csv', index=False)
    catalog_attributes().to_csv('attributes_summary.csv', index=False)
    summarize_crs().to_csv('crs_summary.csv', index=False)
    ph = check_placeholders(parse_metadata())
    if not ph.empty:
        print("Placeholders detected:", ph['LAYER_NAME'].tolist())

    # Inspections & Analysis
    inspect_shapefiles()
    analyze_raster()
    generate_slope_aspect()
    summarize_attributes()

    # Visualizations
    visualize_vectors()
    visualize_dem()
    thematic_maps()

    # Spatial computations (visual only)
    compute_building_volume()
    buildings_near_roads()

    print("Processing complete.")

if __name__ == '__main__':
    main()
