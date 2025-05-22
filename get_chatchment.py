# in this script I want to use the downloaded dem_tiles to compute the flow direction using pysheds
# merge the .tif dem tiles if there are multiple in the folder
# condition the dem with pysheds (fill pits, flat areas, etc.)
# compute the flow direction
# plot the flow direction
# save the flow direction as a .tif file

#%% Import libraries
import os
import numpy as np
import rasterio
import rioxarray
import matplotlib.pyplot as plt
from pathlib import Path
from pysheds.grid import Grid
import rasterio.merge
from rasterio.merge import merge
import geopandas as gpd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import LogNorm
from rasterio.mask import mask
import numpy.ma as ma
from rasterio.features import geometry_mask, rasterize
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
from cartopy.mpl.gridliner import LongitudeFormatter, LatitudeFormatter
import matplotlib.ticker as mticker
import osmnx as ox
from shapely.geometry import box
import pandas as pd
import matplotlib.patches as mpatches
from rasterio.transform import rowcol
from rasterio.transform import xy
from src.vis.utils import create_map_figure
from matplotlib.colors import ListedColormap


def rasterize_osm_water(osm_water, dem_path, output_dir):
    """Rasterize OSM water features to binary raster
    
    Args:
        osm_water (GeoDataFrame): OSM water features
        dem_path (str or Path): Path to the DEM file to get metadata from
        output_dir (str or Path): Directory for saving the water raster
        
    Returns:
        tuple: (numpy.ndarray, Path) Binary water raster and path to saved file
    """
    print("\n=== Starting rasterize_osm_water ===")
    print(f"Rasterizing {len(osm_water)} water features")
    
    # Load metadata from DEM file
    with rasterio.open(dem_path) as src:
        meta = src.meta.copy()
        height = meta['height']
        width = meta['width']
        transform = meta['transform']
    
    print(f"Loaded metadata from DEM: {dem_path}")
    print(f"Raster dimensions: {height} x {width} pixels")
    
    # Rasterize OSM water features
    if not osm_water.empty:
        shapes = [(geom, 1) for geom in osm_water.geometry]
        water_raster = rasterize(
            shapes=shapes,
            out_shape=(height, width),
            transform=transform,
            fill=0,
            all_touched=True,
            dtype=np.uint8
        )
        print(f"Rasterized {len(shapes)} shapes")
        water_pixels = np.sum(water_raster == 1)
        print(f"Water pixels: {water_pixels} ({water_pixels/(height*width)*100:.2f}% of raster)")
    else:
        # Create empty raster if no water features
        water_raster = np.zeros((height, width), dtype=np.uint8)
        print("Created empty water raster (no features)")
    
    # Save raster to output directory
    output_dir = Path(output_dir)
    osm_water_path = output_dir / 'water_raster.tif'
    print(f"Writing OSM water raster to: {osm_water_path}")
    
    with rasterio.open(osm_water_path, 'w', **{
        'driver': 'GTiff',
        'height': height,
        'width': width,
        'count': 1,
        'dtype': np.uint8,
        'crs': meta['crs'],
        'transform': transform,
        'nodata': 0
    }) as dst:
        dst.write(water_raster[np.newaxis, :, :])        

    print("=== Completed rasterize_osm_water ===\n")
    return water_raster, osm_water_path


def get_osm_water(dem_path):
    """Get OSM water features within DEM extent
    
    Args:
        dem_path (str or Path): Path to the DEM file
        
    Returns:
        GeoDataFrame: OSM water features
    """
    print("\n=== Starting get_osm_water ===")
    
    # Get bounds from DEM
    with rasterio.open(dem_path) as src:
        bounds = src.bounds  # left, bottom, right, top
        dem_crs = src.crs
    
    # Create polygon from bounds
    bbox = (bounds.left, bounds.bottom, bounds.right, bounds.top)
    polygon = box(*bbox)
    print(f"DEM bbox: {bbox}")
    
    # Define water-related tags to query
    water_tags = {
        'natural': ['water'],
        'waterway': ['river', 'stream', 'canal', 'drain', 'tidal_channel'],
        'landuse': ['reservoir', 'basin']
    }
    print(f"Using water tags: {water_tags}")
    
    all_features = []
    
    # Query each tag separately
    for category, tags in water_tags.items():
        if isinstance(tags, list):
            for tag in tags:
                tag_dict = {category: tag}
                try:
                    print(f"Querying OSM for {tag_dict}...")
                    gdf = ox.features_from_polygon(polygon, tag_dict)
                    print(f"Retrieved {len(gdf)} features for {tag_dict}")
                    all_features.append(gdf)
                except Exception as e:
                    print(f"Warning: Failed for tags {tag_dict}: {e}")
        else:
            # Handle boolean tags (like waterway=True)
            tag_dict = {category: tags}
            try:
                print(f"Querying OSM for {tag_dict}...")
                gdf = ox.features_from_polygon(polygon, tag_dict)
                print(f"Retrieved {len(gdf)} features for {tag_dict}")
                all_features.append(gdf)
            except Exception as e:
                print(f"Warning: Failed for tags {tag_dict}: {e}")
    
    # Combine all features or create empty GeoDataFrame if none found
    if all_features:
        water_gdf = gpd.GeoDataFrame(pd.concat(all_features, ignore_index=True), crs=4326)
        print(f"Combined {len(water_gdf)} total water features")
    else:
        water_gdf = gpd.GeoDataFrame(geometry=[], crs=4326)
        print("Warning: No OSM water features found in DEM extent")
    
    # Convert to same CRS as DEM
    water_gdf = water_gdf.to_crs(dem_crs)
    
    # Filter to valid geometry types
    valid_types = ['Polygon', 'MultiPolygon', 'LineString', 'MultiLineString']
    water_gdf = water_gdf[water_gdf.geometry.geom_type.isin(valid_types)]
    print(f"After filtering, {len(water_gdf)} valid water features remain")
    print("=== Completed get_osm_water ===\n")
    
    return water_gdf

# Set up directories
dem_tiles_dir = Path("data/dem_tiles")
admin_bounds_dir = Path("data/admin_bound")

output_dir = Path("data/rasters")
plot_dir = Path("data/plots")
water_dir = Path("data/water")
output_dir.mkdir(parents=True, exist_ok=True)
plot_dir.mkdir(parents=True, exist_ok=True)
water_dir.mkdir(parents=True, exist_ok=True)

# some reusable functions


def clip_raster(raster_path, boundary_geom):
    """Clip raster to boundary and return clipped array and extent"""
    # Get the clipped raster
    clipped, transform = mask(
        rasterio.open(raster_path),
        [boundary_geom.__geo_interface__],
        crop=True,
        filled=True,
        nodata=0
    )
    clipped = clipped[0]
    
    # Create masked array
    masked = ma.masked_values(clipped, 0)
    
    # Calculate extent
    height, width = clipped.shape
    left, bottom = transform * (0, height)
    right, top = transform * (width, 0)
    extent = [left, right, bottom, top]
    
    return masked, extent

#%% FATHOM DEM TILES

# Check for DEM tiles
dem_files = list(dem_tiles_dir.glob("*.tif"))
if not dem_files:
    print("No DEM tiles found in data/dem_tiles. Please run get_dem_tiles.py and download the required tiles.")
    import sys
    sys.exit(1)

# Merge DEM tiles if multiple exist
print(f"Found {len(dem_files)} DEM tile(s)")

if len(dem_files) == 1:
    merged_dem_path = dem_files[0]
    print(f"Using single DEM file: {merged_dem_path.name}")
else:
    print(f"Merging {len(dem_files)} DEM tiles...")
    
    # Open all raster files and print metadata of first tile
    src_files_to_mosaic = [rasterio.open(file) for file in dem_files]
    print("\nMetadata of first DEM tile:")
    print(f"Driver: {src_files_to_mosaic[0].meta['driver']}")
    print(f"Data type: {src_files_to_mosaic[0].meta['dtype']}")
    print(f"No data value: {src_files_to_mosaic[0].meta['nodata']}")
    print(f"CRS: {src_files_to_mosaic[0].meta['crs']}")
    print(f"Transform: {src_files_to_mosaic[0].meta['transform']}\n")
    
    # Merge the rasters
    mosaic, out_trans = merge(src_files_to_mosaic)
    
    # Crop the mosaic to the buffered bbox
    print("Cropping merged DEM to buffered bounding box...")
    
    # Load the buffered bbox
    buffered_file = Path("data/buffered_bbox/buffered_bbox.shp")
    if not buffered_file.exists():
        print(f"Buffered bbox file not found: {buffered_file}")
        print("Please run get_admin_bound.py first")
        sys.exit(1)
    
    buffered_gdf = gpd.read_file(buffered_file)
    
    # Get the metadata for the merged raster
    temp_meta = src_files_to_mosaic[0].meta.copy()
    temp_meta.update({
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_trans,
    })
    
    # Create a temporary file for the merged raster
    with rasterio.open("temp_merged.tif", "w", **temp_meta) as temp:
        temp.write(mosaic)
    
    # Open the temporary merged raster and crop it
    with rasterio.open("temp_merged.tif") as src:
        # Transform buffered geometry to match the DEM's CRS if necessary
        buffered_geom = buffered_gdf.to_crs(src.crs).geometry[0]
        
        # Mask the raster with the buffered bbox
        from rasterio.mask import mask
        cropped_mosaic, crop_trans = mask(src, [buffered_geom.__geo_interface__], crop=True)
        
        # Update the mosaic and transform with the cropped values
        mosaic = cropped_mosaic
        out_trans = crop_trans
    
    # Clean up temporary file
    os.remove("temp_merged.tif")
    
    print(f"Cropped DEM shape: {mosaic.shape}")
    
    # Copy the metadata from the first file
    out_meta = src_files_to_mosaic[0].meta.copy()
    
    # Update metadata for the merged file
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_trans,
    })
    # Convert DEM from centimeters to meters
    print(f"DEM elevation range before conversion: {np.nanmin(mosaic):.2f}cm to {np.nanmax(mosaic):.2f}cm")
    mosaic = mosaic / 100  # Convert cm to m
    print(f"DEM elevation range after conversion: {np.nanmin(mosaic):.2f}m to {np.nanmax(mosaic):.2f}m")

    # Write the merged raster
    merged_dem_path = output_dir / "merged_dem.tif"
    with rasterio.open(merged_dem_path, "w", **out_meta) as dest:
        dest.write(mosaic)
    
    # Close all source files
    for src in src_files_to_mosaic:
        src.close()
    
    print(f"Merged DEM saved to {merged_dem_path}")

# Initialize pysheds grid
print(f"Initializing pysheds grid from {merged_dem_path}")
grid = Grid.from_raster(str(merged_dem_path))
dem = grid.read_raster(str(merged_dem_path))

print(f"DEM shape: {dem.shape}")
print(f"DEM resolution: {grid.affine}")


#%% OPEN STREET MAP WATER FEATURES

# load admin boundary and reproject to the same crs as the grid
admin_files = list(admin_bounds_dir.glob("*.shp"))
admin_bound = gpd.read_file(admin_files[0])
admin_shp = ShapelyFeature(Reader(admin_files[0]).geometries(),
                                ccrs.PlateCarree(), edgecolor='black')
admin_bound_proj = admin_bound.to_crs(grid.crs)

# Get the DEM's geographical extent directly from the grid
xmin, xmax, ymin, ymax = grid.extent
dem_bounds_latlon = (xmin, ymin, xmax, ymax)

# Get water features using our new function
print("Fetching water features from OpenStreetMap for the DEM area...")
water_gdf = get_osm_water(str(merged_dem_path))
water_shp = ShapelyFeature(water_gdf.geometry, ccrs.PlateCarree(), edgecolor='blue')
print(f'Number of water features: {len(water_gdf)}')

# save the water features
valid_types = ['Polygon', 'MultiPolygon', 'LineString', 'MultiLineString']
water_gdf = water_gdf[water_gdf.geometry.geom_type.isin(valid_types)]

# Handle potential duplicate column names (case-insensitive)
cols = water_gdf.columns.tolist()
cols_lower = [col.lower() if isinstance(col, str) else col for col in cols]
duplicates = {}

for i, col in enumerate(cols):
    if isinstance(col, str):  # Skip non-string columns like geometry
        lower_col = col.lower()
        if cols_lower.count(lower_col) > 1:
            # If this is a duplicate column (case-insensitive)
            if lower_col not in duplicates:
                duplicates[lower_col] = 0
            else:
                duplicates[lower_col] += 1
                # Rename the column with a suffix
                cols[i] = f"{col}_{duplicates[lower_col]}"

if duplicates:
    print(f"Fixed {sum(duplicates.values())} duplicate column names")
    water_gdf.columns = cols

# Use GeoPackage format to support mixed geometry types
water_file = water_dir / "water_features.gpkg"
water_gdf.to_file(water_file, driver="GPKG")
print(f"Water features saved to {water_file}")

# Rasterize the water features using our function
water_raster, water_raster_file = rasterize_osm_water(water_gdf, str(merged_dem_path), water_dir)

#%% PLOT WATER RASTER
fig, ax = create_map_figure()
# Create a masked array where 0 values are transparent
masked_water = np.ma.masked_where(water_raster == 0, water_raster)
im = ax.imshow(masked_water, cmap='Blues', extent=grid.extent,
                transform=ccrs.PlateCarree(), zorder=1, alpha=1)
ax.set_extent(grid.extent)
plt.title('Water Raster within Admin Boundary (red)')
plt.tight_layout()
plt.savefig(plot_dir / 'water_raster.png')
plt.show()

# plot the water raster within admin boundary
fig, ax = create_map_figure()
cliped_raster, clipped_extent = clip_raster(water_raster_file, admin_bound_proj.geometry.values[0])
# Create a masked array where 0 values are transparent
masked_clipped = np.ma.masked_where(cliped_raster == 0, cliped_raster)
im = ax.imshow(masked_clipped, cmap='Blues', extent=clipped_extent,
                transform=ccrs.PlateCarree(), zorder=1, alpha=1)
ax.set_extent(clipped_extent)
plt.title('Water Raster within Admin Boundary (red)')
plt.tight_layout()
plt.savefig(plot_dir / 'water_raster_clipped.png')
plt.show()
#%% Plot DEM

fig, ax = create_map_figure()

im = ax.imshow(dem, cmap='terrain', extent=dem.extent, transform=ccrs.PlateCarree(), zorder=1)
ax.set_extent(dem.extent)
plt.colorbar(im, ax=ax, label='Elevation (m)')
plt.title('FathomDEM with Admin Boundary (red)')

plt.tight_layout()
plt.savefig(plot_dir / 'original_dem.png')
plt.show()

# plot the clipped DEM
fig, ax = create_map_figure()

dem_masked, clipped_extent = clip_raster(merged_dem_path, admin_bound_proj.geometry.values[0])

im = ax.imshow(dem_masked, cmap='terrain', extent=clipped_extent,
                transform=ccrs.PlateCarree(), zorder=1)
ax.set_extent(clipped_extent)
plt.title('FathomDEM within Admin Boundary (red)')
plt.colorbar(im, ax=ax, label='Elevation (m)', orientation='horizontal')
plt.tight_layout()
plt.savefig(plot_dir / 'original_dem_clipped.png')
plt.show()

#%% Condition DEM
print("Stream burning...")
# Burn the streams in the DEM by lowering elevation by 20m where water features exist
burn_depth = 20  # meters
# Make a copy of the original DEM before burning
dem_original = dem.copy()
# Get water feature mask (1 where water exists, 0 elsewhere)
water_mask = water_raster == 1
# Apply the burn by subtracting the depth where water features exist
dem[water_mask] -= burn_depth
print(f"Burned {np.sum(water_mask)} stream cells by {burn_depth}m")

print("Filling pits...")
dem_filled = grid.fill_pits(dem)
print("Filling depressions...")
dem_filled = grid.fill_depressions(dem_filled)
print("Resolving flats...")
dem_flooded = grid.resolve_flats(dem_filled)
print("Done")
#%% Plot conditioned DEM
from src.vis.utils import create_map_figure
fig, ax = create_map_figure()

im = ax.imshow(dem_flooded, cmap='terrain', extent=grid.extent,
                transform=ccrs.PlateCarree(), zorder=1)
ax.set_extent(grid.extent)
plt.colorbar(im, ax=ax, label='Elevation (m)')
plt.title('Conditioned DEM with Admin Boundary (red)')
plt.tight_layout()
plt.savefig(plot_dir / 'conditioned_dem.png')
plt.show()

# Get metadata from the input file for saving rasters
with rasterio.open(merged_dem_path) as src:
    meta = src.meta.copy()

# save the conditioned DEM
conditioned_dem_file = output_dir / "conditioned_dem.tif"
with rasterio.open(conditioned_dem_file, 'w', **meta) as dst:
    dst.write(dem_flooded.astype(np.float32), 1)

# plot the clipped conditioned DEM
fig, ax = create_map_figure()
clipped_dem, clipped_extent = clip_raster(conditioned_dem_file, admin_bound_proj.geometry.values[0])
im = ax.imshow(clipped_dem, cmap='terrain', extent=clipped_extent,
                transform=ccrs.PlateCarree(), zorder=1)
ax.set_extent(clipped_extent)
plt.colorbar(im, ax=ax, label='Elevation (m)', orientation='horizontal')
plt.title('Conditioned DEM within Admin Boundary (red)')
plt.tight_layout()
plt.savefig(plot_dir / 'conditioned_dem_clipped.png')
plt.show()

print("Flow direction computation complete")

#%% Compute flow direction
print("Computing flow direction...")
# D8 flow directions: 1=E, 2=SE, 4=S, 8=SW, 16=W, 32=NW, 64=N, 128=NE

flow_dir = grid.flowdir(dem_flooded, routing='d8', flats=-1, pits=-2, nodata_out=0)
print(f"Flow direction min: {np.min(flow_dir)}")
print(f"Flow direction max: {np.max(flow_dir)}")
print(f"Flow direction number of flats: {np.sum(flow_dir == -1)}")
print(f"Flow direction number of pits: {np.sum(flow_dir == -2)}")
print(f"Flow direction number of nodata: {np.sum(flow_dir == 0)}")

#%% Plot and saveflow direction
fig, ax = create_map_figure()
im = ax.imshow(flow_dir, cmap='viridis', extent=grid.extent,
                transform=ccrs.PlateCarree(), zorder=1)
ax.set_extent(grid.extent)
plt.colorbar(im, ax=ax, label='Flow Direction')
plt.title('Flow Direction (D8) with Admin Boundary (red)')

plt.tight_layout()
plt.savefig(plot_dir / 'flow_direction.png')
plt.show()


flow_dir_file = output_dir / "flow_direction.tif"

# Get metadata from the input file
with rasterio.open(merged_dem_path) as src:
    meta = src.meta.copy()

# Update metadata for flow direction raster
meta.update({
    'dtype': rasterio.uint8,
    'nodata': 0
})

# Write flow direction raster
print(f"Saving flow direction to {flow_dir_file}...")
with rasterio.open(flow_dir_file, 'w', **meta) as dst:
    dst.write(flow_dir.astype(np.uint8), 1)

# plot the clipped flow direction
fig, ax = create_map_figure()
flow_dir_clipped, clipped_extent = clip_raster(flow_dir_file, admin_bound_proj.geometry.values[0])

im = ax.imshow(flow_dir_clipped, cmap='viridis', extent=clipped_extent,
                transform=ccrs.PlateCarree(), zorder=1)
ax.set_extent(clipped_extent)
plt.colorbar(im, ax=ax, label='Flow Direction', orientation='horizontal')
plt.title('Flow Direction within Admin Boundary (red)')
plt.tight_layout()
plt.savefig(plot_dir / 'flow_direction_clipped.png')
plt.show()


# %% Compute accumulation
acc = grid.accumulation(flow_dir)

# Plot full flow accumulation
fig, ax = create_map_figure()
im = ax.imshow(acc, extent=grid.extent, cmap='cubehelix', transform=ccrs.PlateCarree(),
                norm=LogNorm(1, acc.max()), interpolation='bilinear', zorder=1)
ax.set_extent(grid.extent)
plt.colorbar(im, ax=ax, label='Upstream Cells')
plt.title('Flow Accumulation with Admin Boundary (red)')
plt.tight_layout()
plt.savefig(plot_dir / 'flow_accumulation_full.png')
plt.show()

# save the full flow accumulation
acc_file = output_dir / "flow_accumulation.tif"
with rasterio.open(acc_file, 'w', **meta) as dst:
    dst.write(acc.astype(np.float32), 1)


# Plot clipped data
fig, ax = create_map_figure()
acc_clipped, clipped_extent = clip_raster(acc_file, admin_bound_proj.geometry.values[0])
im = ax.imshow(acc_clipped,
                zorder=3,
                extent=clipped_extent, 
                cmap='cubehelix',
                transform=ccrs.PlateCarree(),
                norm=LogNorm(1, acc_clipped.max()),
                interpolation='bilinear')
ax.set_extent(clipped_extent)
plt.colorbar(im, ax=ax, label='Upstream Cells', orientation='horizontal')
plt.title('Flow Accumulation within Admin Boundary (red)')

plt.tight_layout()
plt.savefig(plot_dir / 'flow_accumulation_clipped.png')
plt.show()

#%% Compute Stream Network /(Accumulation > 1000)
stream_network = acc > 1000

#Plot Stream Network
fig, ax = create_map_figure()
im = ax.imshow(np.where(stream_network, stream_network, np.nan), cmap='bone', extent=grid.extent,
                transform=ccrs.PlateCarree(), zorder=1)
ax.set_extent(grid.extent)
#plt.colorbar(im, ax=ax, label='Stream Network')
plt.title('Streams (Accumulation > 1000) with Admin Boundary (red)')

plt.tight_layout()
plt.savefig(plot_dir / 'stream_network.png')
plt.show()

# save the stream network
stream_network_file = output_dir / "stream_network.tif"
with rasterio.open(stream_network_file, 'w', **meta) as dst:
    dst.write(stream_network.astype(np.uint8), 1)

# plot the clipped stream network
fig, ax = create_map_figure()
stream_network_clipped, clipped_extent = clip_raster(stream_network_file, admin_bound_proj.geometry.values[0])
im = ax.imshow(np.where(stream_network_clipped, stream_network_clipped, np.nan), cmap='bone', extent=clipped_extent,
                transform=ccrs.PlateCarree(), zorder=1)
ax.set_extent(clipped_extent)
#plt.colorbar(im, ax=ax, label='Stream Network')
plt.title('Streams (Accumulation > 1000) within Admin Boundary (red)')

plt.tight_layout()
plt.savefig(plot_dir / 'stream_network_clipped.png')
plt.show()


# %% find the water cell within the admin boundary with the highest accumulation

# Create a mask for points that are inside the admin boundary
print("Finding highest accumulation point within admin boundary...")

# First attempt with clipped raster
cliped_acc, clipped_extent = clip_raster(acc_file, admin_bound_proj.geometry.values[0])
acc_max = np.nanmax(cliped_acc)
acc_max_coords = np.where(cliped_acc == acc_max)
# Check if any valid coordinates were found
if len(acc_max_coords[0]) > 0:
    acc_max_coords = (acc_max_coords[0][0], acc_max_coords[1][0])
else:
    raise ValueError("No valid maximum accumulation point found within admin boundary")

print(f"Initial highest accumulation point at coordinates (row, col): {acc_max_coords}")
print(f"Maximum accumulation value: {acc_max}")

# Get the geographic coordinates of the maximum accumulation cell
# Read the transform from the accumulation file
with rasterio.open(acc_file) as src:
    transform = src.transform
    height, width = src.shape

x, y = transform * (acc_max_coords[1] + 0.5, acc_max_coords[0] + 0.5)

print(f"Geographic coordinates (x, y): ({x}, {y})")

# Verify if this point is truly inside the admin boundary polygon
from shapely.geometry import Point
# Create a proper mask for inside the polygon
from rasterio.features import geometry_mask

# Get the geometry of the admin boundary
geom = [admin_bound_proj.geometry.values[0].__geo_interface__]

# Create a mask where True represents cells INSIDE the admin boundary
boundary_mask = geometry_mask(geom, (height, width), transform, invert=True)

# Apply the mask to the original accumulation data
acc_within_boundary = acc.copy()
acc_within_boundary[~boundary_mask] = 0  # Set values outside boundary to 0

# Find the maximum accumulation within the boundary
acc_max = np.max(acc_within_boundary)
if acc_max == 0:
    print("No accumulation points found within the admin boundary!")
else:
    acc_max_coords = np.where(acc_within_boundary == acc_max)
    acc_max_coords = (acc_max_coords[0][0], acc_max_coords[1][0])
    print(f"Found new highest accumulation point within boundary at (row, col): {acc_max_coords}")
    print(f"New maximum accumulation value: {acc_max}")
    
    # Get new coordinates
    row, col = acc_max_coords
    x, y = transform * (col + 0.5, row + 0.5)  # Add 0.5 to get center of pixel
    
    # Double-check this point is inside
    point = Point(x, y)
    is_inside = admin_bound_proj.geometry.values[0].contains(point)
    print(f"Is new point inside admin boundary polygon? {is_inside}")


#%% plot the cell with the highest accumulation with the admin boundary
fig, ax = create_map_figure()
im = ax.imshow(np.where(water_raster, water_raster, np.nan), cmap='bone', extent=grid.extent,
                transform=ccrs.PlateCarree(), zorder=1)
ax.set_extent(grid.extent)
# Plot the highest accumulation point
point = ax.scatter(x, y, transform=ccrs.PlateCarree(), color='blue', s=300, marker='*', 
          zorder=3, edgecolor='black', label='Max Flow Acc')

plt.title('Highest Flow Accumulation with Admin Boundary (red)')

water_patch = mpatches.Patch(color='black', label='OSM Water')
ax.legend(handles=[water_patch, point], loc='lower left', facecolor='none', edgecolor='none')

plt.tight_layout()
plt.savefig(plot_dir / 'highest_acc_cell.png')
plt.show()

# plot the cell with the highest accumulation within the admin boundary
fig, ax = create_map_figure()
cliped_water_raster, clipped_extent = clip_raster(water_raster_file, admin_bound_proj.geometry.values[0])
im = ax.imshow(np.where(cliped_water_raster, cliped_water_raster, np.nan), cmap='bone', extent=clipped_extent,
                transform=ccrs.PlateCarree(), zorder=1)
ax.set_extent(clipped_extent)
# Plot the highest accumulation point
point = ax.scatter(x, y, transform=ccrs.PlateCarree(), color='blue', s=300, marker='*', 
          zorder=3, edgecolor='black', label='Max Flow Acc')

plt.title('Highest Flow Accumulation within Admin Boundary (red)')

water_patch = mpatches.Patch(color='black', label='OSM Water')
ax.legend(handles=[water_patch, point], loc='lower left', facecolor='none', edgecolor='none')

plt.tight_layout()
plt.savefig(plot_dir / 'highest_acc_cell_clipped.png')
plt.show()


# %%Export the highest accumulation point as a shapefile
point_geom = Point(x, y)
point_gdf = gpd.GeoDataFrame({
    'type': ['highest'],
    'accumulation': [acc_max],
    'geometry': [point_geom]
}, crs=grid.crs)

# Save as shapefile
point_file = output_dir / "flow_acc_points.shp"
point_gdf.to_file(point_file)
print(f"Flow accumulation points saved to {point_file}")



#%% Compute Catchment
print(f"Computing catchment from pour point {x}, {y}")

grid = Grid.from_raster(str(conditioned_dem_file))
dem = grid.read_raster(str(conditioned_dem_file))

pit_filled_dem = grid.fill_pits(dem)
pits = grid.detect_pits(pit_filled_dem)
if pits.any():
    print("ERROR: Pits still remain in the DEM after pit filling")
    print(f"Number of remaining pits: {np.sum(pits)}")
    raise ValueError("DEM still contains pits after pit filling")

flooded_dem = grid.fill_depressions(pit_filled_dem)
depressions = grid.detect_depressions(flooded_dem)
if depressions.any():
    print("ERROR: Depressions still remain in the DEM after depression filling")
    print(f"Number of remaining depressions: {np.sum(depressions)}")
    raise ValueError("DEM still contains depressions after depression filling")

inflated_dem = grid.resolve_flats(flooded_dem)
flats = grid.detect_flats(inflated_dem)
if flats.any():
    print("ERROR: Flats still remain in the DEM after flat resolution")
    print(f"Number of remaining flats: {np.sum(flats)}")
    raise ValueError("DEM still contains flats after flat resolution")

print("Conditioned DEM without errors")

flow_dir = grid.flowdir(inflated_dem)
acc = grid.accumulation(flow_dir)
x_snap, y_snap = grid.snap_to_mask(acc > 1000, (x, y))
print(f"Pouring point coordinates: {x}, {y}")
print(f"Snapped coordinates: {x_snap}, {y_snap}")

catchment = grid.catchment(x=x_snap, y=y_snap, fdir=flow_dir, xytype='coordinate')

grid.clip_to(catchment)
catchment = grid.view(catchment)
# Debug catchment
print(f"Catchment shape: {catchment.shape}")
print(f"Catchment data type: {catchment.dtype}")
print(f"Catchment min: {np.min(catchment)}, max: {np.max(catchment)}")
print(f"Number of catchment cells: {np.sum(catchment == 1)}")

# Debug grid information
print(f"Grid affine transform: {grid.affine}")
print(f"Grid CRS: {grid.crs}")

# Calculate cell size from the affine transform
# The cell size in the x-direction is the absolute value of element [0]
# The cell size in the y-direction is the absolute value of element [4]
cell_size_x = abs(grid.affine[0])
cell_size_y = abs(grid.affine[4])
cell_area = cell_size_x * cell_size_y  # in square meters

# Calculate catchment area
catchment_area_cells = np.sum(catchment == 1)
catchment_area_m2 = catchment_area_cells * cell_area
catchment_area_km2 = catchment_area_m2 / 1_000_000  # Convert to km²

print(f"Cell size: {cell_size_x} x {cell_size_y} meters")
print(f"Cell area: {cell_area} m²")
print(f"Catchment area: {catchment_area_m2:.2f} m² = {catchment_area_km2:.2f} km²")

# Save the catchment to a file
catchment_file = output_dir / "catchment.tif"
with rasterio.open(catchment_file, 'w', **meta) as dst:
    dst.write(catchment.astype(np.uint8), 1)
print(f"Catchment area saved to {catchment_file}")

# Verify the catchment file was created
if os.path.exists(catchment_file):
    print(f"Catchment file exists at {catchment_file}")
    with rasterio.open(catchment_file) as src:
        print(f"Catchment file shape: {src.shape}")
        catchment_data = src.read(1)
        print(f"Catchment file values - min: {catchment_data.min()}, max: {catchment_data.max()}")
else:
    print(f"WARNING: Catchment file does not exist at {catchment_file}")

# %% Plot the catchment
fig, ax = create_map_figure()

# Load the catchment raster
catchment_raster = rasterio.open(catchment_file)
catchment_data = catchment_raster.read(1)

# Use imshow instead of contour for binary data
catchment_color = '#64B5CD'
# Create custom colormap with a single color
catchment_cmap = ListedColormap([catchment_color])

# Plot the catchment with proper parameters
im = ax.imshow(np.where(catchment_data == 1, 1, np.nan), 
               extent=grid.extent,
               zorder=3, 
               cmap=catchment_cmap,  # Use cmap instead of color
               alpha=0.7, 
               transform=ccrs.PlateCarree())

# Add highest accumulation point
point = ax.scatter(x, y, transform=ccrs.PlateCarree(), color='blue', s=300, marker='*', 
          zorder=4, edgecolor='black', label='Pour Point')

ax.set_extent(dem.extent)
# Create legend elements
admin_color = '#C44E51'
buffered_color = '#8C8C8C'
catchment_patch = mpatches.Patch(color=catchment_color, label='Catchment', alpha=0.7)
admin_patch = plt.scatter([], [], c=admin_color, marker='s', s=200, label='Admin Boundary', edgecolor='none')
#buffered_patch = plt.scatter([], [], c=buffered_color, marker='s', s=200, label='Buffered Bounding Box', edgecolor='none')

# Combine all legend elements
legend_elements = [catchment_patch, point, admin_patch]
ax.legend(handles=legend_elements, loc='lower left', edgecolor='none', facecolor='white')

plt.title(f'Catchment (catchment area: {catchment_area_cells} cells, {catchment_area_km2:.2f} km²) with Admin Boundary (red)')

plt.tight_layout()
plt.savefig(plot_dir / 'catchment.png')
plt.show()

# %% Plot the clipped catchment
fig, ax = create_map_figure()

# clipped catchment
catchment_clipped, clipped_extent = clip_raster(catchment_file, admin_bound_proj.geometry.values[0])

# Use imshow instead of contour for binary data
# Plot the catchment with proper parameters
im = ax.imshow(np.where(catchment == 1, 1, np.nan), 
               extent=grid.extent,
               zorder=3, 
               cmap=catchment_cmap,  # Use cmap instead of color
               alpha=0.7, 
               transform=ccrs.PlateCarree())

# Add highest accumulation point
point = ax.scatter(x, y, transform=ccrs.PlateCarree(), color='blue', s=300, marker='*', 
          zorder=4, edgecolor='black', label='Pour Point')

ax.set_extent(clipped_extent)

# Create legend elements
admin_color = '#C44E51'
buffered_color = '#8C8C8C'
catchment_patch = mpatches.Patch(color=catchment_color, label='Catchment', alpha=0.7)
admin_patch = plt.scatter([], [], c=admin_color, marker='s', s=200, label='Admin Boundary', edgecolor='none')
#buffered_patch = plt.scatter([], [], c=buffered_color, marker='s', s=200, label='Buffered Bounding Box', edgecolor='none')

# Combine all legend elements
legend_elements = [catchment_patch, point, admin_patch]
ax.legend(handles=legend_elements, loc='lower left', edgecolor='none', facecolor='white')
plt.title(f'Catchment within Admin Boundary (red)')

plt.tight_layout()
plt.savefig(plot_dir / 'catchment_clipped.png')
plt.show()





# %%
