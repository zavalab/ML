import geopandas as gpd

from HydroGraph_functions import *

# Load in the dataframes of Wisconsin waterbodies and rivers
WILakes  = gpd.GeoDataFrame.from_file("WIgeodataframes/lakes/lakes.shp")
WIRivers = gpd.GeoDataFrame.from_file("WIgeodataframes/rivers/rivers.shp")

print("Adding lake/river node columns")

# Add the lake and river node columns to the dataframes
WIRivers, WILakes = add_lake_river_node_column(WIRivers.copy(), WILakes.copy())

# Save the new geodataframes to a pickle file; because the geodataframes contain
# lists, they cannot be saved to a shp file
WILakes.to_pickle("WILakes.df")
WIRivers.to_pickle("WIRivers.df")


