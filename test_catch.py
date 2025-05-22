
#%%
from pysheds.grid import Grid
dem_path = "data/rasters/merged_dem.tif"

grid = Grid.from_raster(dem_path)
dem = grid.read_raster(dem_path)

# Plot the raw DEM
fig, ax = plt.subplots(figsize=(8,6))
fig.patch.set_alpha(0)

plt.imshow(grid.view(dem), cmap='terrain', zorder=1)
plt.colorbar(label='Elevation (m)')
plt.title('Digital elevation map', size=14)
plt.tight_layout()


# %%
pits = grid.detect_pits(dem)
# Plot pits
fig, ax = plt.subplots(figsize=(8,6))
fig.patch.set_alpha(0)

plt.imshow(pits, cmap='Greys_r', zorder=1)
plt.title('Pits', size=14)
plt.tight_layout()

# Fill pits
pit_filled_dem = grid.fill_pits(dem)
pits = grid.detect_pits(pit_filled_dem)
assert not pits.any()

#%%
depressions = grid.detect_depressions(pit_filled_dem)
# Plot depressions
fig, ax = plt.subplots(figsize=(8,6))
fig.patch.set_alpha(0)

plt.imshow(depressions, cmap='Greys_r', zorder=1)
plt.title('Depressions', size=14)
plt.tight_layout()
# Fill depressions
flooded_dem = grid.fill_depressions(pit_filled_dem)
depressions = grid.detect_depressions(flooded_dem)
assert not depressions.any()

# %%
flats = grid.detect_flats(flooded_dem)
# Plot flats
fig, ax = plt.subplots(figsize=(8,6))
fig.patch.set_alpha(0)

plt.imshow(flats, cmap='Greys_r', zorder=1)
plt.title('Flats', size=14)
plt.tight_layout()
inflated_dem = grid.resolve_flats(flooded_dem)
flats = grid.detect_flats(inflated_dem)
assert not flats.any()

plt.imshow(flats, cmap='Greys_r', zorder=1)
plt.title('Flats after resolution', size=14)
plt.tight_layout()
inflated_dem = grid.resolve_flats(flooded_dem)
flats = grid.detect_flats(inflated_dem)

depressions = grid.detect_depressions(pit_filled_dem)
# Plot depressions
fig, ax = plt.subplots(figsize=(8,6))
fig.patch.set_alpha(0)

plt.imshow(depressions, cmap='Greys_r', zorder=1)
plt.title('Depressions after resolution', size=14)
plt.tight_layout()
# Fill depressions
flooded_dem = grid.fill_depressions(pit_filled_dem)
depressions = grid.detect_depressions(flooded_dem)
assert not depressions.any()
# %%
import matplotlib.colors as colors
# Compute flow direction based on corrected DEM
fdir = grid.flowdir(inflated_dem)


# Compute flow accumulation based on computed flow direction
acc = grid.accumulation(fdir)

fig, ax = plt.subplots(figsize=(8,6))
fig.patch.set_alpha(0)
im = ax.imshow(acc, zorder=2,
               cmap='cubehelix',
               norm=colors.LogNorm(1, acc.max()),
               interpolation='bilinear')
plt.colorbar(im, ax=ax, label='Upstream Cells')
plt.title('Flow Accumulation', size=14)
plt.tight_layout()

#%%
x, y = 6.700833333333334, 51.16722222222222
x_snap, y_snap = grid.snap_to_mask(acc > 1000, (x, y))

catch = grid.catchment(x=x_snap, y=y_snap, fdir=fdir, xytype='coordinate')

grid.clip_to(catch)
catch_view = grid.view(catch)
#%%
import matplotlib.pyplot as plt
import numpy as np
fig, ax = plt.subplots(figsize=(8,6))
fig.patch.set_alpha(0)

plt.grid('on', zorder=0)
im = ax.imshow(np.where(catch_view, catch_view, np.nan), extent=grid.extent,
               zorder=1, cmap='Greys_r')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Delineated Catchment', size=14)



# %%
from src.vis.utils import create_map_figure
grid.viewfinder = fdir.viewfinder

fig, ax = create_map_figure()
im = ax.imshow(np.where(catch_view, catch_view, np.nan), extent=grid.extent,
               zorder=3, cmap='Greys_r')
ax.scatter(x, y, color='red', zorder=4)
ax.set_extent(grid.extent)

plt.show()

# %%
