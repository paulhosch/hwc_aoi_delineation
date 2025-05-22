# Leverkusen AOI Delineation

Workflow to delineate the catchment for a Pourpoint defined as the cell with the highest flow accumulation within a queried administrative boundary.

## 1 Query Admin Bounds from OpenstreetMap and buffer bbox

![admin_bound](https://github.com/user-attachments/assets/a6c703f9-5098-4b32-90f9-4da04ae7187e)

## 2 Process FathomDEM

![original_dem](https://github.com/user-attachments/assets/04aea21b-44d2-424f-95cd-af644a7ef978)
![original_dem_clipped](https://github.com/user-attachments/assets/7b36af9a-d270-42b9-a0da-a94c0001b81c)

## 3 Query Water from Open StreetMap and rasterize

![water_raster](https://github.com/user-attachments/assets/56b06a15-dada-4ceb-9dbe-8079d69776df)
![water_raster_clipped](https://github.com/user-attachments/assets/598e572a-71eb-4626-8e40-3d997f76abfd)

## 4 Stream Burn and Condition DEM

![conditioned_dem](https://github.com/user-attachments/assets/1a131bb2-2c2c-435e-89e6-745342c3302a)
![conditioned_dem_clipped](https://github.com/user-attachments/assets/6ccf3fd6-fb76-4a0e-92a8-de8d2b236391)

## 5 Compute Flow Direction

![flow_direction](https://github.com/user-attachments/assets/acc89af2-9bfc-4c67-9c2f-e514d3f5dfdb)
![flow_direction_clipped](https://github.com/user-attachments/assets/94164457-87ef-4180-b117-3fb8be6c21f5)

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
