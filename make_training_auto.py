#!/usr/bin/env python3
"""
make_training_auto.py

Auto-generate training polygons from Sentinel-2 multiband TIFF using:
 - SCL (if present)
 - NDVI, NDWI, NDBI indices
 - Optional OSM rivers shapefile to force water

Outputs:
 - data/training/training_polygons_auto.geojson
 - outputs/maps/training_polygons_preview.png

Edit PATHS below if your project root is different.
"""
import os
import json
import numpy as np
import rasterio
from rasterio.features import shapes, rasterize
import geopandas as gpd
from shapely.geometry import shape
from skimage.morphology import remove_small_objects, closing, disk
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter("ignore")

# ---------------------------
# CONFIG - EDIT IF NEEDED
# ---------------------------
# If you saved the repo in another folder, change REPO_ROOT to that path
REPO_ROOT = os.path.join(os.path.expanduser("~"), "turing-geospatial-analyst-portfolio")
PROJECT = os.path.join(REPO_ROOT, "python_geospatial", "project03_landcover_classification")

SENTINEL = os.path.join(PROJECT, "data", "raw", "sentinel2_denpasar_multiband.tif")
OSM_RIVERS = os.path.join(PROJECT, "data", "raw", "Denpasar_rivers.shp")  # optional
OUT_GEOJSON = os.path.join(PROJECT, "data", "training", "training_polygons_auto.geojson")
PREVIEW_PNG = os.path.join(PROJECT, "outputs", "maps", "training_polygons_preview.png")

# Processing params (tweak if needed)
MIN_AREA_M2 = 2000        # minimum polygon area to keep (meters^2). reduce if CRS is degrees.
SMALL_OBJ_PIXELS = 50     # remove small objects (in pixel units)
MORPH_RADIUS = 3          # morphological closing radius (pixels)
NDVI_VEG_TH = 0.30        # NDVI threshold for vegetation
NDVI_BARE_TH = 0.12       # NDVI below this often bare
NDBI_URBAN_TH = 0.08      # NDBI threshold for urban (tune)
WATER_NDWI_TH = 0.1       # NDWI > this likely water
WATER_NDBI_TH = -0.15     # require low NDBI for water

# ---------------------------
# Ensure folders exist
# ---------------------------
for p in [os.path.join(PROJECT, "data", "training"), os.path.join(PROJECT, "outputs", "maps")]:
    os.makedirs(p, exist_ok=True)

# ---------------------------
# Helper functions
# ---------------------------
def autoscale(band):
    """If band values are in 0..10000 scale, convert to 0..1"""
    band = band.astype('float32')
    mx = np.nanmax(band)
    if mx > 2:
        return band / 10000.0
    return band

def clean_bool(arr_bool, morph_radius=MORPH_RADIUS, min_size=SMALL_OBJ_PIXELS):
    """Apply closing + remove small speckles"""
    if not np.any(arr_bool):
        return arr_bool
    arrc = closing(arr_bool, disk(morph_radius))
    arrc = remove_small_objects(arrc.astype(bool), min_size=min_size)
    return arrc

def approx_area_m2(geom, crs):
    """If CRS is geographic (EPSG:4326) approximate area using local meters per deg"""
    area = geom.area
    if crs and '4326' in str(crs):
        lon, lat = geom.representative_point().x, geom.representative_point().y
        meters_per_deg = 111320 * np.cos(np.deg2rad(lat))
        return area * (meters_per_deg**2)
    return area

# ---------------------------
# Read Sentinel TIFF and compute indices
# ---------------------------
if not os.path.exists(SENTINEL):
    raise FileNotFoundError(f"Sentinel TIFF not found: {SENTINEL}\nPlace your multiband TIFF into PROJECT/data/raw/")

print("Opening sentinel:", SENTINEL)
with rasterio.open(SENTINEL) as src:
    meta = src.meta.copy()
    transform = src.transform
    crs = src.crs
    h, w = src.height, src.width
    count = src.count
    print(f"Raster size: {w} x {h}, bands: {count}, crs: {crs}")

    # Expect band order: B2,B3,B4,B8,B11,B12, (optional SCL as next band)
    if count < 6:
        raise ValueError("Expected >=6 bands (B2,B3,B4,B8,B11,B12). Found: {}".format(count))

    b2 = src.read(1).astype('float32')
    b3 = src.read(2).astype('float32')
    b4 = src.read(3).astype('float32')
    b8 = src.read(4).astype('float32')
    b11 = src.read(5).astype('float32')
    b12 = src.read(6).astype('float32') if count >= 6 else np.zeros((h,w), dtype='float32')
    scl = src.read(7).astype('int16') if count >= 7 else None

# autoscale
b2 = autoscale(b2); b3 = autoscale(b3); b4 = autoscale(b4); b8 = autoscale(b8); b11 = autoscale(b11); b12 = autoscale(b12)

# compute indices
ndvi = (b8 - b4) / (b8 + b4 + 1e-8)
ndwi = (b3 - b8) / (b3 + b8 + 1e-8)
ndbi = (b11 - b8) / (b11 + b8 + 1e-8)

# valid mask (avoid zeros / nans)
valid = np.isfinite(b2) & np.isfinite(b3) & np.isfinite(b4) & np.isfinite(b8)
valid &= ((b2 != 0) | (b3 != 0) | (b4 != 0))

# initial masks
water_mask = np.zeros((h,w), dtype=bool)
veg_mask = np.zeros((h,w), dtype=bool)
bare_mask = np.zeros((h,w), dtype=bool)

if scl is not None:
    # SCL codes common mapping: 6=water, 4=vegetation, 5=not-vegetated
    water_mask |= (scl == 6)
    veg_mask |= (scl == 4)
    bare_mask |= (scl == 5)
    print("SCL present: water pixels:", int(water_mask.sum()), "veg pixels:", int(veg_mask.sum()))

# index-based masks (complement SCL)
veg_mask |= (ndvi >= NDVI_VEG_TH) & valid
bare_mask |= (ndvi < NDVI_BARE_TH) & valid
urban_mask = (ndbi > NDBI_URBAN_TH) & (ndvi < 0.25) & valid
water_mask |= ((ndwi > WATER_NDWI_TH) & (ndbi < WATER_NDBI_TH) & valid)

# remove cloud/shadow if SCL exists
if scl is not None:
    cloud_codes = [3,8,9,10]  # cloud-shadow and cloud
    mask_cloud = np.isin(scl, cloud_codes)
    veg_mask &= ~mask_cloud
    urban_mask &= ~mask_cloud
    bare_mask &= ~mask_cloud
    water_mask &= ~mask_cloud

# combine with priority: water > urban > veg > bare
label = np.zeros((h,w), dtype='uint8')
label[water_mask] = 1
label[(label==0) & urban_mask] = 2
label[(label==0) & veg_mask] = 3
label[(label==0) & bare_mask] = 4

print("Raw label counts:", {i:int((label==i).sum()) for i in [1,2,3,4]})

# optional: use OSM rivers to force water (if available)
if os.path.exists(OSM_RIVERS):
    print("Applying OSM rivers (found):", OSM_RIVERS)
    try:
        rivers = gpd.read_file(OSM_RIVERS)
        if rivers.crs is None:
            rivers.set_crs(crs, inplace=True)
        # rasterize rivers (all_touched True to cover line width)
        shapes_riv = ((geom, 1) for geom in rivers.geometry)
        riv_arr = rasterize(shapes_riv, out_shape=(h,w), transform=transform, fill=0, all_touched=True, dtype='uint8')
        label[riv_arr==1] = 1
        print("Applied rivers mask -> water forced.")
    except Exception as e:
        print("OSM rivers rasterize error:", e)

# cleaning: morphology and remove small objects per-class
lab_clean = np.zeros_like(label)
for cls in [1,2,3,4]:
    m = (label == cls)
    if not np.any(m):
        continue
    mc = clean_bool(m, morph_radius=MORPH_RADIUS, min_size=SMALL_OBJ_PIXELS)
    lab_clean[mc] = cls

# fallback: keep original label where cleaning removed everything
lab_clean[lab_clean==0] = label[lab_clean==0]

print("After cleaning counts:", {i:int((lab_clean==i).sum()) for i in [1,2,3,4]})

# polygonize shapes (only keep shapes>MIN_AREA_M2)
pixel_area = abs(transform.a * transform.e)  # pixel width * pixel height (may be degrees or meters)
print("Pixel area (raw transform units^2):", pixel_area)

polys = []
count_all = 0
for geom, val in shapes(lab_clean, mask=lab_clean>0, transform=transform):
    count_all += 1
    geom_shape = shape(geom)
    area_m2 = approx_area_m2(geom_shape, crs)
    if area_m2 >= MIN_AREA_M2:
        polys.append({"geometry": geom_shape, "class": int(val), "area_m2": float(area_m2)})

print("Polygons produced (after area filter):", len(polys), "from raw shapes:", count_all)

if len(polys) == 0:
    # Try relaxed filter once
    print("No polygons survived. Relaxing MIN_AREA_M2 and SMALL_OBJ_PIXELS and re-run quick poly.")
    polys = []
    for geom, val in shapes(lab_clean, mask=lab_clean>0, transform=transform):
        geom_shape = shape(geom)
        area_m2 = approx_area_m2(geom_shape, crs)
        # accept anything > 200 m^2
        if area_m2 >= 200:
            polys.append({"geometry": geom_shape, "class": int(val), "area_m2": float(area_m2)})
    print("Relaxed-run polygons:", len(polys))
    if len(polys) == 0:
        raise RuntimeError("No polygons generated even after relaxation. Check input TIFF and parameters.")

gdf = gpd.GeoDataFrame(polys, crs=crs)
gdf['geometry'] = gdf.geometry.buffer(0)  # fix invalids if any

# Save GeoJSON
gdf.to_file(OUT_GEOJSON, driver='GeoJSON')
print("Saved training polygons:", OUT_GEOJSON)
print("Class counts (output):", gdf['class'].value_counts().to_dict())

# Quick preview: RGB + boundaries
try:
    # create quick normalized RGB for display
    rgb = np.dstack([np.clip(b4,0,1), np.clip(b3,0,1), np.clip(b2,0,1)])
    fig, ax = plt.subplots(1,1, figsize=(9,9))
    ax.imshow(rgb, origin='upper')
except Exception:
    fig, ax = plt.subplots(1,1, figsize=(9,9))
    ax.imshow(ndvi, cmap='RdYlGn', origin='upper')

# color mapping
color_map = {1:'#2b83ba', 2:'#d7191c', 3:'#1a9850', 4:'#fdae61'}
for cls, grp in gdf.groupby('class'):
    try:
        grp.boundary.plot(ax=ax, edgecolor=color_map.get(cls,'k'), linewidth=1, label=f'{cls} ({len(grp)})')
    except Exception:
        pass

ax.legend()
ax.axis('off')
ax.set_title("Auto training polygons preview")
plt.savefig(PREVIEW_PNG, dpi=180, bbox_inches='tight')
plt.show()
print("Preview saved to:", PREVIEW_PNG)

# Final short QA output
print("\n--- Quick QA ---")
print("PROJECT:", PROJECT)
print("SENTINEL:", SENTINEL)
print("OUTPUT GEOJSON:", OUT_GEOJSON)
print("PREVIEW PNG:", PREVIEW_PNG)
print("Number polygons:", len(gdf))
print("Classes:", gdf['class'].value_counts().to_dict())
