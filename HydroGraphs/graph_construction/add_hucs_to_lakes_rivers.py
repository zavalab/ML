import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point
import warnings
from HydroGraph_functions import *

warnings.filterwarnings('ignore')

# Load in the base dataframes of Wisconsin waterbodies and rivers
WILakes  = gpd.GeoDataFrame.from_file("WIgeodataframes/lakes/lakes.shp")
WIRivers = gpd.GeoDataFrame.from_file("WIgeodataframes/rivers/rivers.shp")

# Load in the HUC8, HUC10, and HUC12 shapefiles for Wisconsin
HUC8  = gpd.GeoDataFrame.from_file("WIgeodataframes/HUC8/HUC8.shp")
HUC10 = gpd.GeoDataFrame.from_file("WIgeodataframes/HUC10/HUC10.shp")
HUC12 = gpd.GeoDataFrame.from_file("WIgeodataframes/HUC12/HUC12.shp") 

print("Adding HUCs to lakes")

# Add the HUC8, HUC10, and HUC12 unit codes to the lakes and rivers
# These are determined by what HUC the centroid of the lake lies in
WILakes = add_HUC8(WILakes.copy(), HUC8.copy())
WILakes = add_HUC10(WILakes.copy(), HUC10.copy())
WILakes = add_HUC12(WILakes.copy(), HUC12.copy())

print("Adding HUCs to rivers")

WIRivers = add_HUC8(WIRivers.copy(), HUC8.copy())
WIRivers = add_HUC10(WIRivers.copy(), HUC10.copy())
WIRivers = add_HUC12(WIRivers.copy(), HUC12.copy())


# Save the lakes and rivers
WILakes.to_file("WIgeodataframes/lakes/lakes.shp")
WIRivers.to_file("WIgeodataframes/rivers/rivers.shp")
