import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LongitudeFormatter, LatitudeFormatter
import cartopy.crs as ccrs
import rasterio
from rasterio.mask import mask
import numpy as np
import numpy.ma as ma
from pathlib import Path
from cartopy.feature import ShapelyFeature
from cartopy.io.shapereader import Reader

def create_map_figure():
    """Create a figure with gridlines and admin boundary feature"""

    buffered_file = Path("data/buffered_bbox/buffered_bbox.shp")
    admin_file = Path("data/admin_bound/admin_bound.shp")

    admin_shp = ShapelyFeature(Reader(admin_file).geometries(),
                                    ccrs.PlateCarree(), edgecolor='black')
    buffered_shp = ShapelyFeature(Reader(buffered_file).geometries(),
                                    ccrs.PlateCarree(), edgecolor='blue')

    fig = plt.figure(figsize=(9, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Directly add the ShapelyFeature objects to the map
    admin_color = '#C44E51'
    ax.add_feature(admin_shp, zorder=2, facecolor='none', edgecolor=admin_color, linewidth=2)

    if buffered_shp is not None:
        buffered_color = '#8C8C8C'
        ax.add_feature(buffered_shp, zorder=1, facecolor=buffered_color, edgecolor='none')
        
        # Extract bounds from the first geometry in the ShapelyFeature
        # ShapelyFeature doesn't have bounds, but its geometries do
        geoms = list(buffered_shp.geometries())
        if geoms:
            bounds = geoms[0].bounds  # minx, miny, maxx, maxy
            ax.set_extent(bounds)

    gl = ax.gridlines(draw_labels=["left", "bottom"],
                  xformatter=LongitudeFormatter(number_format='.1f'),
                  yformatter=LatitudeFormatter(number_format='.1f'),
                  zorder=0)
    gl.xlocator = mticker.LinearLocator(3)    
    gl.ylocator = mticker.LinearLocator(3)

    # add legend to bootom left 
    legend_elements = [
        plt.scatter([], [], c=admin_color, marker='s', s=200, label='Admin Boundary', edgecolor='none'),
        plt.scatter([], [], c=buffered_color, marker='s', s=200, label='Buffered Bounding Box', edgecolor='none')
    ]
    ax.legend(handles=legend_elements, loc='lower left', edgecolor='none', facecolor='white')
    
    plt.tight_layout()
    return fig, ax

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