# Leverkusen AOI Delineation

Workflow to delineate the catchment for a Pourpoint defined as the cell with the highest flow accumulation within a queried administrative boundary.

## 1 Query Admin Bounds from OpenstreetMap and buffer bbox

![admin_bound](https://github.com/user-attachments/assets/a6c703f9-5098-4b32-90f9-4da04ae7187e)

## 2 Process FathomDEM
![original_dem](https://github.com/user-attachments/assets/b5fc6011-adc2-41d0-8afa-fbea4a8ff2b2)
![original_dem_clipped](https://github.com/user-attachments/assets/61109905-9d5c-494a-b331-5339fda33031)

## 3 Query Water from Open StreetMap and rasterize

![water_raster](https://github.com/user-attachments/assets/56b06a15-dada-4ceb-9dbe-8079d69776df)
![water_raster_clipped](https://github.com/user-attachments/assets/598e572a-71eb-4626-8e40-3d997f76abfd)

## 4 Stream Burn and Condition DEM
![conditioned_dem](https://github.com/user-attachments/assets/7367b178-5ab0-451e-8798-77acad62a085)
![conditioned_dem_clipped](https://github.com/user-attachments/assets/d0c373b8-ee2f-4578-8dbd-3a79387f8d78)


## 5 Compute Flow Direction
![flow_direction](https://github.com/user-attachments/assets/fb39d59c-f16c-4aae-92aa-671c6857ed98)
![flow_direction_clipped](https://github.com/user-attachments/assets/7c0a78a8-e17c-443c-90a2-4b7501ef5ea0)

## 6 Compute Flow Accumulation

![flow_accumulation_full](https://github.com/user-attachments/assets/0c75d27f-81fc-4c6d-b392-663809e5088e)
![flow_accumulation_clipped](https://github.com/user-attachments/assets/7c991435-715a-4023-9c89-a32fa460fabd)

## 7 Compute Stream Network

![stream_network](https://github.com/user-attachments/assets/0992d026-29c7-4587-a4f1-df7d9d2b66c4)
![stream_network_clipped](https://github.com/user-attachments/assets/1c033132-bf4e-4447-a3e5-7971687dd228)

## 8 Determine Pour Point as Highest Accumulation Cell

![highest_acc_cell](https://github.com/user-attachments/assets/a3a87a08-2aff-4c72-be76-b997c127c801)
![highest_acc_cell_clipped](https://github.com/user-attachments/assets/ff2b5dab-f5dd-4f5d-bb9f-c36611b2bf20)

## 9 Delineate Catchment

![catchment](https://github.com/user-attachments/assets/fbaff545-658c-42ce-aab8-868901019cea)
![catchment_clipped](https://github.com/user-attachments/assets/e61f8193-afc7-4c2a-a3f1-363921c214aa)

## Data

The in and output data can be downloaded from:
https://rwth-aachen.sciebo.de/s/UycaRu8bONFMggI

## Data Sources

- **FathomDEM**: Global terrain map [Zenodo API]

  - Source: Uhe, P., Lucas, C., Hawker, L., et al. (2025). FathomDEM: an improved global terrain map using a hybrid vision transformer model. Environmental Research Letters, 20(3).
  - Resolution: 30m

- **OpenStreetMap**: Water features [OSM API]
  - Source: © OpenStreetMap contributors
  - Features:Rivers, lakes, and other water bodies
