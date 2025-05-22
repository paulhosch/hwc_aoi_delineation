
#%% Import libraries
import os
import math
import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path

#%% Set up directories
admin_bounds_dir = Path("data/admin_bounds")
dem_tiles_dir = Path("data/dem_tiles")
dem_tiles_dir.mkdir(parents=True, exist_ok=True)

#%% Get list of available boundary files
boundary_files = list(admin_bounds_dir.glob("*.shp"))
if not boundary_files:
    print("No boundary files found in data/admin_bounds. Please run get_admin_bound.py first.")
else:
    print("Available boundary files (using the first one):")
    for i, file in enumerate(boundary_files):
        print(f"{i+1}. {file.name}")
    
    # Use the first boundary file
    boundary_file = boundary_files[0]
    
    # Load boundary file
    print(f"Loading {boundary_file}...")
    aoi_gdf = gpd.read_file(boundary_file)
    
    #    Ensure AOI is in lat/lon (EPSG:4326)
    aoi_gdf = aoi_gdf.to_crs(epsg=4326)
    
    # Get the bounding box
    min_lon, min_lat, max_lon, max_lat = aoi_gdf.total_bounds
    print(f"Bounding box: lon({min_lon:.4f}, {max_lon:.4f}), lat({min_lat:.4f}, {max_lat:.4f})")
    
    # Determine integer tile indices
    lat_start = math.floor(min_lat)
    lat_end = math.floor(max_lat)
    lon_start = math.floor(min_lon)
    lon_end = math.floor(max_lon)
    
    # Generate list of required tiles
    tile_names = []
    for lat in range(lat_start, lat_end + 1):
        for lon in range(lon_start, lon_end + 1):
            lat_prefix = 'n' if lat >= 0 else 's'
            lon_prefix = 'e' if lon >= 0 else 'w'
            tile_filename = f"{lat_prefix}{abs(lat):02d}{lon_prefix}{abs(lon):03d}.tif"
            tile_names.append(tile_filename)
    
    # Print required tiles
    print(f"Required tiles: {', '.join(tile_names)}")
    
    # Check which tiles are already downloaded
    existing_tiles = list(dem_tiles_dir.glob("*.tif"))
    existing_tile_names = [file.name for file in existing_tiles]
    
    # Print download instructions
    missing_tiles = [tile for tile in tile_names if tile not in existing_tile_names]
    
    if not missing_tiles:
        print("All required tiles are already downloaded!")
    else:
        print(f"\nPlease download the following {len(missing_tiles)} tiles manually:")
        for tile in missing_tiles:
            print(f"- {tile}")
        print("\nTiles can be downloaded from FathomDEM or similar DEM repositories.")
        print("After downloading, place them in the data/dem_tiles directory.")
# %%
