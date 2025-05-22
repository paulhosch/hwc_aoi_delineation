import rasterio
from rasterio.features import shapes
import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path
from rasterio.features import geometry_mask, rasterize
from shapely.geometry import box

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