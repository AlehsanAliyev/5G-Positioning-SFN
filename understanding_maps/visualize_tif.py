import numpy as np
import rasterio
import matplotlib.pyplot as plt

# Open slope and aspect
with rasterio.open('slope.tif') as ds_slope, rasterio.open('aspect.tif') as ds_aspect:
    slope = ds_slope.read(1)
    aspect = ds_aspect.read(1)
    nodata = ds_slope.nodata

# Mask: only show aspect where slope > 1 degree
mask = (slope != nodata) & (slope > 1)
aspect_masked = np.where(mask, aspect, np.nan)

# Plot slope
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
im0 = ax[0].imshow(slope, cmap='viridis')
ax[0].set_title('Slope (degrees)')
fig.colorbar(im0, ax=ax[0], label='Slope°')

# Plot masked aspect
im1 = ax[1].imshow(aspect_masked, cmap='hsv', vmin=0, vmax=360)
ax[1].set_title('Aspect (only where slope > 1°)')
fig.colorbar(im1, ax=ax[1], label='Aspect°')

plt.tight_layout()
plt.show()
