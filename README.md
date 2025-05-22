# Leverkusen AOI Delineation

A geospatial toolkit for delineating areas of interest (AOI) for hydrological analysis.

## Setup

1. Clone this repository
2. Run the setup script to create a virtual environment and register it as an IPython kernel:

```bash
python setup_env.py
```

3. Launch Jupyter Notebook or JupyterLab
4. Open the Python scripts and select the `lev_aoi_env` kernel

## Workflow

1. Run `get_admin_bound.py` to download administrative boundaries
2. Run `get_dem_tiles.py` to determine required DEM tiles
3. Download the required DEM tiles and place them in `data/dem_tiles/`
4. Run `compute_flow_dir.py` to calculate flow direction

## Dependencies

All dependencies are specified in `requirements.txt` and will be installed by the setup script.
