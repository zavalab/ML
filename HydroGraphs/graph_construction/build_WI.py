import numpy as np
import geopandas as gpd
import pandas as pd


WI_counties = gpd.GeoDataFrame.from_file("Counties/County_Boundaries_24K.shp")
WI_counties = WI_counties.to_crs("EPSG:4326")

WI_counties["d"] = 0
WI = WI_counties.dissolve(by="d")

WI.to_file("lakes_rivers/WI")

