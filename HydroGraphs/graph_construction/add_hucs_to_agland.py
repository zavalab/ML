import geopandas as gpd
from HydroGraph_functions import *
import warnings
import pandas as pd
from tqdm import tqdm
warnings.filterwarnings('ignore')

# Read in the agricultural land file
agland = gpd.GeoDataFrame.from_file("agland/WisconsinFieldBoundaries2019.shp")

# Convert crs
agland = agland.to_crs("EPSG:4326")

print("loaded in land cover")

# Load in HUC shapefiles
HUC8  = gpd.GeoDataFrame.from_file("WIgeodataframes/HUC8/HUC8.shp")
HUC10 = gpd.GeoDataFrame.from_file("WIgeodataframes/HUC10/HUC10.shp")
HUC12 = gpd.GeoDataFrame.from_file("WIgeodataframes/HUC12/HUC12.shp") 

print("loaded in HUCs")

# Add huc codes for the agricultural land
agland["huc8"]  = 0
agland["huc10"] = 0
agland["huc12"] = 0

# Define a place to save the clipped data
df_huc8 = pd.DataFrame(columns = agland.columns)

# Clip all the data by HUC8 watershed; this makes sure that, when agricultural land
# is assigned to a watershed, none of the land is outside of that watershed
for i in tqdm(range(len(HUC8))):
    # Define the HUC8 polygon for clipping
    mask = HUC8.geometry.iloc[i]

    # Clip the agricultural land by HUC8 watershed
    clipping = gpd.clip(agland, mask)
    
    # Set the HUC8 watershed code
    clipping["huc8"] = HUC8.huc8.iloc[i]
    
    # Store the clipped data
    df_huc8 = df_huc8.append(clipping)

# Convert the clipped data to a GeoDataFrame
agland = gpd.GeoDataFrame(df_huc8, crs = "EPSG:4326").reset_index(drop = True)

print("finished huc8 clipping")

# Define a place to save the clipped data for HUC10 watersheds
df_huc10 = pd.DataFrame(columns = agland.columns)

# Clip all the data by HUC10 watershed
for i in tqdm(range(len(HUC10))):
    # Define the HUC10 polygon for clipping
    mask = HUC10.geometry.iloc[i]

    # Define the HUC10 and HUC8 codes
    huc10_val = HUC10.huc10.iloc[i]
    huc8_val  = HUC10.huc8.iloc[i]

    # Only get the agricultural land in the HUC8 watershed; this reduces 
    # computational time rather than clipping form the entire agland shapefile
    huc8_ag_land = agland[agland.huc8 == huc8_val]

    # Clip the agricultural land by HUC10 watershed
    clipping = gpd.clip(huc8_ag_land, mask)
    
    # Set the HUC10 watershed code
    clipping["huc10"] = huc10_val
    
    # Store the clipped data
    df_huc10 = df_huc10.append(clipping)

# Convert the clipped data to a GeoDataFrame
agland = gpd.GeoDataFrame(df_huc10, crs = "EPSG:4326").reset_index(drop = True)

print("finished huc10 clipping")

# Define a place to save the clipped data for HUC12 watersheds
df_huc12 = pd.DataFrame(columns = agland.columns)

# Clip all the data by HUC12 watershed
for i in tqdm(range(len(HUC12))):
    # Deinfe the HUC12 polygon for clipping
    mask = HUC12.geometry.iloc[i]

    # Define the HUC10 and HUC12 codes
    huc12_val = HUC12.huc12.iloc[i]
    huc10_val = HUC12.huc10.iloc[i]

    # Only get the agricultural alnd in the HUC10 watershed
    huc10_ag_land = agland[agland.huc10 == huc10_val]

    # Clip the agricultural land by HUC12 watershed
    clipping = gpd.clip(huc10_ag_land, mask)
    
    # Set the HUC12 watershed code
    clipping["huc12"] = huc12_val
    
    # Store the clipped data
    df_huc12 = df_huc12.append(clipping)

# Convert the clipped data to a GeoDataFrame
agland = gpd.GeoDataFrame(df_huc12, crs = "EPSG:4326").reset_index(drop = True)

print("finished huc12 clipping")

# Save the agricultural land file
agland.to_pickle("agland.df")