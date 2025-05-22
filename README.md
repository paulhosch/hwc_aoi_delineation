# Leverkusen AOI Delineation

Workflow to delineate the catchment for a Pourpoint defined as the cell with the highest flow accumulation within a queried administrative boundary.

## 1 Query Admin Bounds from OpenstreetMap and buffer bbox

<img src="https://github.com/user-attachments/assets/9fd72e15-9100-4417-b3c5-f74cc4f66b9b" width="90%" />

## 2 Process FathomDEM

<p float="left">
  <img src="https://github.com/user-attachments/assets/680dc997-94e4-45d2-ba5c-1737fce34a69" width="45%" />
  <img src="https://github.com/user-attachments/assets/3f5a0310-3b3e-4a1d-86d6-8401a6cec852" width="45%" />
</p>

## 3 Query Water from Open StreetMap and rasterize

<p float="left">
  <img src="https://github.com/user-attachments/assets/45f2bacd-e318-48fd-a358-3e4803fec4aa" width="45%" />
  <img src="https://github.com/user-attachments/assets/5c359994-5b40-404b-abbf-7dbc04de9de6" width="45%" />
</p>

## 4 Stream Burn and Condition DEM

<p float="left">
  <img src="https://github.com/user-attachments/assets/fb264e70-8dc4-43a4-a3cd-e564236b3ca2" width="45%" />
  <img src="https://github.com/user-attachments/assets/839b0327-8cb8-48eb-9439-d77c82cbf2e2" width="45%" />
</p>

## 5 Compute Flow Direction

<p float="left">
  <img src="https://github.com/user-attachments/assets/3ef28ebe-d9d4-4914-8a47-425429ea69c4" width="45%" />
  <img src="https://github.com/user-attachments/assets/e7fad4c1-6ecb-4349-bdd4-371eb2f5b4bd" width="45%" />
</p>

## 6 Compute Flow Accumulation

<p float="left">
  <img src="https://github.com/user-attachments/assets/25b8e872-7295-40d2-bb61-033a8da5d69b" width="45%" />
  <img src="https://github.com/user-attachments/assets/18dcb506-3c8f-4743-ba93-7baeba72df6b" width="45%" />
</p>

## 7 Compute Stream Network

<p float="left">
  <img src="https://github.com/user-attachments/assets/c7d4d12e-fb07-439f-b3d8-5137c3e176ba" width="45%" />
  <img src="https://github.com/user-attachments/assets/3d2f2f4a-b65c-41f2-95f0-370cd3e490c6" width="45%" />
</p>

## 8 Determine Pour Point as Highest Accumulation Cell

<p float="left">
  <img src="https://github.com/user-attachments/assets/95b81dbf-43fb-4ffa-9d9b-8ed533206f07" width="45%" />
  <img src="https://github.com/user-attachments/assets/7e0157b8-f56b-4895-bb4b-fc2f537bccff" width="45%" />
</p>

## 9 Delineate Catchment

<p float="left">
  <img src="https://github.com/user-attachments/assets/03ca0dd9-ef85-420c-a162-924a5d5992f2" width="45%" />
  <img src="https://github.com/user-attachments/assets/1ac6a0de-6689-4c0d-860b-b39694ed9221" width="45%" />
</p>
