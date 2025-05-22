# Leverkusen AOI Delineation

Workflow to delineate the catchment for a Pourpoint defined as the cell with the highest flow accumulation within a queried administrative boundary.

## 1 Query Admin Bounds from OpenstreetMap and buffer bbox
![admin_bound](https://github.com/user-attachments/assets/e99e7c86-a622-416e-9732-046e8dbce7ad)


## 2 Process FathomDEM
![original_dem](https://github.com/user-attachments/assets/eb2646c5-b599-4039-b4b7-76d6fce31155)
![original_dem_clipped](https://github.com/user-attachments/assets/8cd04363-9882-4215-a6ea-ad211558c071)


## 3 Query Water from Open StreetMap and rasterize
![water_raster](https://github.com/user-attachments/assets/73d0fab3-bc7d-4093-850e-a1f009cee9da)
![water_raster_clipped](https://github.com/user-attachments/assets/2c85de74-9276-4c60-81ff-ebedfe2ed22e)


## 4 Stream Burn and Condition DEM
![conditioned_dem](https://github.com/user-attachments/assets/26d680b7-fb74-4d1c-993a-43167a743d00)
![conditioned_dem_clipped](https://github.com/user-attachments/assets/76f23fe9-d2b0-44e7-a8bf-b5bae70588ce)


## 5 Compute Flow Direction
![flow_direction](https://github.com/user-attachments/assets/1599d1c3-d29c-4b2a-a17e-596ae32583fe)
![flow_direction_clipped](https://github.com/user-attachments/assets/725de937-39f2-45c1-86a4-e97f791b5348)


## 6 Compute Flow Accumulation
![flow_accumulation_full](https://github.com/user-attachments/assets/93f1aa3d-4273-43b9-9196-04a74a987eb8)
![flow_accumulation_clipped](https://github.com/user-attachments/assets/c180dfa5-a5ac-4b0e-9264-78e625a2df3e)


## 7 Compute Stream Network
![stream_network](https://github.com/user-attachments/assets/74b309df-022a-4cdd-ac7d-3bc9e85060ee)
![stream_network_clipped](https://github.com/user-attachments/assets/4707d8b1-e0b9-4db5-8408-2137d72d9f60)


## 8 Determine Pour Point as Highest Accumulation Cell
![highest_acc_cell](https://github.com/user-attachments/assets/f45c0d6a-e988-455b-8915-c113d51e8251)
![highest_acc_cell_clipped](https://github.com/user-attachments/assets/de5f84ce-876a-4ea2-9828-5f8e7832ed8b)


## 9 Delineate Catchment
![catchment](https://github.com/user-attachments/assets/43dc3f09-2a7b-4c73-aa8c-e3339401c754)
![catchment_clipped](https://github.com/user-attachments/assets/883d89f3-9ab3-4efd-883f-24aed1a728a0)

