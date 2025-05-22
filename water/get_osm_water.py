import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.features import shapes
from shapely.geometry import box
import osmnx as ox

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