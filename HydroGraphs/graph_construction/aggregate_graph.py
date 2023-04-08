import warnings

import pandas as pd
import numpy as np
import geopandas as gpd
import pandas as pd
from HydroGraph_functions import *
warnings.filterwarnings('ignore')

# Read in the Wisconsin dataframes for lakes and rivers
WILakes  = pd.read_pickle("WILakes.df")
WIRivers = pd.read_pickle("WIRivers.df")

# Read in the list of tofroms that includes waterbodies
WItofroms_lakes = pd.read_csv("WIgeodataframes/WItofroms_lakes.csv")

# Aggregate the intermediate river nodes
WItofroms_agg = aggregate_river_nodes(WItofroms_lakes, WIRivers, WILakes)

# Save the aggregated list of tofroms
WItofroms_agg.to_csv("WIgeodataframes/WItofroms_agg.csv")
