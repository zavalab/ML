import geopandas as gpd
from geopandas.tools import sjoin
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import re
from tqdm import tqdm
import shapely

lc = pd.read_pickle("raster_to_gdf.df")

lc = lc[lc["raster_val"] != 0]

print("loaded in land cover")

HUC8  = gpd.GeoDataFrame.from_file("../WIgeodataframes/HUC8/HUC8.shp")
HUC10 = gpd.GeoDataFrame.from_file("../WIgeodataframes/HUC10/HUC10.shp")
HUC12 = gpd.GeoDataFrame.from_file("../WIgeodataframes/HUC12/HUC12.shp") 

HUC8  = HUC8.to_crs("EPSG:3071")
HUC10 = HUC10.to_crs("EPSG:3071")
HUC12 = HUC12.to_crs("EPSG:3071")

print("loaded in hucs")

lc["huc8"]  = 0
lc["huc10"] = 0
lc["huc12"] = 0

df = pd.DataFrame(columns = lc.columns)

for i in tqdm(range(len(HUC8))):
    mask = HUC8.geometry.iloc[i]

    clipping = gpd.clip(lc, mask)
    
    clipping["huc8"] = HUC8.huc8.iloc[i]
    
    df = df.append(clipping)

gdf = gpd.GeoDataFrame(df, crs = "EPSG:3071").reset_index(drop = True)

print("DONE WITH HUC8")

df = pd.DataFrame(columns = lc.columns)

for i in tqdm(range(len(HUC10))):
    mask = HUC10.geometry.iloc[i]

    huc10_val = HUC10.huc10.iloc[i]
    huc8_val  = HUC10.huc8.iloc[i]

    huc8_lc = gdf[gdf.huc8 == huc8_val]

    clipping = gpd.clip(huc8_lc, mask)
    
    clipping["huc10"] = huc10_val
    
    df = df.append(clipping)


gdf = gpd.GeoDataFrame(df, crs = "EPSG:3071").reset_index(drop = True)

print("DONE WITH HUC10")

df = pd.DataFrame(columns = lc.columns)

for i in tqdm(range(len(HUC12))):
    mask = HUC12.geometry.iloc[i]

    huc12_val = HUC12.huc12.iloc[i]
    huc10_val = HUC12.huc10.iloc[i]

    huc10_lc = gdf[gdf.huc10 == huc10_val]

    clipping = gpd.clip(huc10_lc, mask)
    
    clipping["huc12"] = huc12_val
    
    df = df.append(clipping)

gdf = gpd.GeoDataFrame(df, crs = "EPSG:3071").reset_index(drop = True)
gdf.to_pickle("land_cover.df")

print("DONE WITH HUC12")
