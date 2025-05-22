#%% set up 
import os
import geopandas as gpd
import osmnx as ox
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import math
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LongitudeFormatter, LatitudeFormatter
from src.vis.utils import create_map_figure
from cartopy.feature import ShapelyFeature
from cartopy.io.shapereader import Reader
import cartopy.crs as ccrs

city = 'Leverkusen'
country = 'Germany'

admin_dir = Path("data/admin_bound")
buffered_dir = Path("data/buffered_bbox")

admin_dir.mkdir(parents=True, exist_ok=True)
buffered_dir.mkdir(parents=True, exist_ok=True)

#%% get the boundary from OpenStreetMap
print(f"Downloading boundary for {city}, {country}...")
# Use the correct function for current OSMnx versions
boundary = ox.geocode_to_gdf(f"{city}, {country}")

#%% process the boundary
# Check if we got results
if boundary.empty:
    print(f"No administrative boundaries found for {city}, {country}")
else:
    print(f"Found {len(boundary)} boundary features")
    
    # Filter to relevant admin boundaries if needed
    # Keep only relevant admin levels (typically 8 for cities)
    if 'admin_level' in boundary.columns:
        boundary = boundary[boundary['admin_level'].isin(['8', '6'])]
        
    # Dissolve boundaries if multiple features exist
    if len(boundary) > 1:
        print(f"Dissolving {len(boundary)} boundary features into one")
        boundary_dissolved = boundary.dissolve()
    else:
        boundary_dissolved = boundary
    
    # Save as shapefile
    admin_file = admin_dir / "admin_bound.shp"
    boundary_dissolved.to_file(admin_file)
    print(f"Saved boundary to {admin_file}")

# %% buffer the boundary by 50% in x and 50% in y and save to data/buffered_bbox/buffered_bbox.shp
# Get the bounding box
bbox = boundary_dissolved.bounds.iloc[0]
min_x, min_y, max_x, max_y = bbox

# Calculate buffer distances (50% in each direction)
width = max_x - min_x
height = max_y - min_y
buffer_x = width * 0.5
buffer_y = height * 0.5

# Create buffered bounding box
buffered_min_x = min_x - buffer_x
buffered_min_y = min_y - buffer_y
buffered_max_x = max_x + buffer_x
buffered_max_y = max_y + buffer_y

# Round to nearest decimal place (floor for min, ceil for max)
buffered_min_x = math.floor(buffered_min_x * 10) / 10
buffered_min_y = math.floor(buffered_min_y * 10) / 10
buffered_max_x = math.ceil(buffered_max_x * 10) / 10
buffered_max_y = math.ceil(buffered_max_y * 10) / 10

print(f"Rounded buffered bbox: ({buffered_min_x}, {buffered_min_y}) to ({buffered_max_x}, {buffered_max_y})")

# Create a polygon from the buffered bbox
from shapely.geometry import box
buffered_bbox_poly = box(buffered_min_x, buffered_min_y, buffered_max_x, buffered_max_y)
buffered_bbox = gpd.GeoDataFrame(geometry=[buffered_bbox_poly], crs=boundary_dissolved.crs)

# Save the buffered bbox
buffered_file = buffered_dir / "buffered_bbox.shp"
buffered_bbox.to_file(buffered_file)
print(f"Saved buffered bounding box to {buffered_file}")

#%% Plot the buffered bbox and admin boundary


# read the saved shapefile

fig, ax = create_map_figure()

ax.set_extent([buffered_min_x, buffered_max_x, buffered_min_y, buffered_max_y])
plt.title(f"Admin Boundary and Buffered Bounding Box: {city}, {country}")
plt.show()
# %%
