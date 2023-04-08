# HydroGraphs

This repository contains the code for the manuscript "A Graph-Based Modeling Framework for Tracing Hydrological Pollutant Transport in Surface Waters." There are three main folders containing code and data, and these are outlined below. We call the framework for building a graph of these hydrological systems "HydroGraphs".

Several of the datafiles for building this framework are large and cannot be stored on Github. To conserve space, the notebook `get_and_unpack_data.ipynb` or the script `get_and_unpack_data.py` can be used to download the data from the Watershed Boundary Dataset (WBD), the National Hydrography Dataset (NHDPlusV2), and the agricultural land dataset for the state of Wisconsin. The files `WILakes.df` and `WIRivers.df` mentioned in section 1 below are contained within the `WI_lakes_rivers.zip` folder, and the files 24k Hydro Waterbodies dataset are contained in a zip file under the directory `DNR_data/Hydro_Waterbodies`. These files can also be unpacked by running the corresponding cells in the notebook `get_and_unpack_data.ipynb` or `get_and_unpack_data.py`. 

## 1. graph_construction
This folder contains the data and code for building a graph of the watershed-river-waterbody hydrological system. It uses data from the Watershed Boundary Dataset (link [here](https://apps.nationalmap.gov/downloader/#/)) and the National Hydrography Dataset (link [here](https://www.epa.gov/waterdata/get-nhdplus-national-hydrography-dataset-plus-data)) as a basis and builds a list of directed edges. We use `NetworkX` to build and visualize the list as a graph. This folder contains the following subfolders:

* `CAFOS_shp` - contains the shape files for concentrated animal feeding operations (CAFOs) in Wisconsin
* `Counties` - contains the shape files for all the counties in Wisconsin, obtained from [here](https://data-wi-dnr.opendata.arcgis.com/datasets/wi-dnr::county-boundaries-24k/about). This shapefile was dissolved to get a polygon of the state of Wisconsin. 
* `agland` - contains a shapefile of the agricultural land for the state of Wisconsin obtained from [here](https://doi.org/10.15482/USDA.ADC/1520625). This file was altered to clip the land by watershed and include the HUC8, HUC10, and HUC12 codes for the watershed that each polgyon of land lies within. The resulting GeoDataFrame is called `agland.df`.
* `lakes_rivers` - contains the basic dataframes obtained from the NHDPlusV2 and WBD. These are contained within the `Watershed4` and `Watershed7` subfolders. These subfolders also contain the `PlusFlow.dbf` file from the NHDPlusV2, and the `PlusFlow.csv` file that is a CSV format of the `PlusFlow.dbf`. In addition, there is a subfolder, `WI`, that contains the shapefile of Wisconsin that was constructed from the Wisconsin counties. 
* `WIgeodataframes` - contains the shapefiles of the lakes, rivers, and HUCs for the state of Wisconsin after running the script `run_WI_graph_code.py` (the resulting files are not included in the repository due to size). These were obtained from the NHDPlusV2 and WBD data in `lakes_rivers`, but with some minor attribute additions and only over the state of Wisconsin. In addition, this folder contains the CSVs of TOCOMID and FROMCOMID list for different graphs. These lists are manipulations of the `PlusFlow.csv` where:
    * `WItofroms.csv` is the `PlusFlow.csv` applied to just the state of Wisconsin. This can be used to form the river graph
    * `WItofroms_lakes.csv` is the `WItofroms.csv` with waterbodies added to the list, replacing some river COMIDs
    * `WItofroms_agg.csv` is the aggregated form of `WItofroms_lakes.csv`

This folder also contains several files and scripts. These are as follows:

* `WILakes.df` - This is a set of the waterbodies from the NHDPlusV2 for the state of Wisconsin. It includes columns for the HUC8, HUC10, and HUC12 codes and a column for intersecting rivers. This file is created by the script `add_river_lake_nodes.py`.
* `WIRivers.df` - This is a set of the rivers from the NHDPlusV2 for the state of Wisconsin. It includes columns for the HUC8, HUC10, and HUC12 codes and a column for intersecting waterbodies. This file is created by the script `add_river_lake_nodes.py`.
* `agland.df` - This is the agricultural land with the HUC8, HUC10, and HUC12 codes added; this file is created by the script `add_hucs_to_agland`.
* `WI_graph_functions.py` - This script contains several functions for working with "Hydrology Graphs." It also contains several functions that are called to form the desired graphs from the NHDPlusV2 and WBD.
* `run_WI_graph_code.py` - This script runs the following other scripts, in the following order, to take the data from `lakes_rivers` and construct the files in `WIgeodataframes` and the files `WILakes.df` and `WIRivers.df`
    * `build_base_dataframes.py` - takes the NHDPlusV2 data and removes the areas outside of Wisconsin.
    * `add_hucs_to_lakes_rivers.py` - adds the HUC8, HUC10, and HUC12 codes to the river and waterbody GeoDataFrames
    * `add_river_lake_nodes.py` - adds the column of intersecting waterbodies to the river GeoDataFrame and adds the column of intersecting rivers to the waterbody GeoDataFrame. 
    * `add_to_comid_list.py` - builds the first two "tofrom" CSVs in the `WIgeodataframes` folder
    * `aggregate_graph.py` - aggregates the graph and forms the `WItofroms_agg.csv`
* `Visualizations.ipynb` - This notebook builds some of the visualizations formed in the manuscript mentioned above, and it contains a test of connectivity to compare the aggregated graph to the original graph. 

## 2. case_studies

This folder contains three .ipynb files for three separate case studies. These three case studies focus on how "Hydrology Graphs" can be used to analyze pollutant impacts in surface waters. Details of these case studies can be found in the manuscript above.

## 3. DNR_data

This folder contains data from the Wisconsin Department of Natural Resources (DNR) on water quality in several Wisconsin lakes. The data was obtained from [here](https://dnr.wi.gov/lakes/waterquality/) using the file `Web_scraping_script.py`. The original downloaded reports are found in the folder `original_lake_reports`. These reports were then cleaned and reformatted using the script `DNR_data_filter.ipynb`. The resulting, cleaned reports are found in the `Lakes` folder. Each subfolder of the `Lakes` folder contains data for a single lake. The two .csvs `lake_index_WBIC.csv` contain an index for what lake each numbered subfolder corresponds. In addition, we added the corresponding COMID in `lake_index_WBIC_COMID.csv` by matching the NHDPlusV2 data to the Wisconsin DNR's 24k Hydro Waterbodies dataset which we downloaded from [here](https://data-wi-dnr.opendata.arcgis.com/datasets/31f1f67253074ef9afe46cd905bff07a/explore?location=44.938463%2C-89.279700%2C7.33). The DNR's reported data only matches lakes to a waterbody identification code (WBIC), so we use HYDROLakes (indexed by WBIC) to match to the COMID. This is done in the `DNR_data_filter.ipynb` script as well. 

## Python Versions

The .py files in `graph_construction/` were run using Python version 3.9.7. The scripts used the following packages and version numbers:
 
* geopandas (0.10.2)
* shapely (1.8.1.post1)
* tqdm (4.63.0)
* networkx (2.7.1)
* pandas (1.4.1)
* numpy (1.21.2)

The .ipynb files in `graph_construction/` and `case_studies` were run using Python version 3.8.5 with the following packages and version numbers:

* geopandas (0.8.2)
* shapely (1.7.1)
* tqdm (4.50.2)
* networkx (2.5)
* matplotlib (3.5.2)
* pandas (1.4.2)

The .py file in `DNR_data` used for webscraping used Python version 3.9.5 with Selenium version 3.141.0. Note that the user must download the webdriver and put in the proper path to run the code. For our results, we used the Chromedriver for Chrome version 92. 

## General Framework

While the code in this repository contains scripts for the state of Wisconsin, the overall framework can easily be applied to other geographical areas as long as the data is in the right format. As our data comes directly from the NHDPlusV2 and WBD datasets, this is easy to set up for other areas. The script `run_WI_graph_code.py` calls 5 other python files that incorporate this framework. In addition, it calls many predefined functions from the file `WI_graph_functions.py`. These functions are a primary contribution of this work. 

The first file called is `build_base_dataframes.py`. This file is primarily data processing; the state of Wisconsin includes two different datasets from the NHDPlusV2 (HUC04 and HUC07), and we needed to first merge these two datasets together and then choose only the waterbodies/rivers/watersheds that are in the state of Wisconsin. For users who want to see an example of how to merge multiple HUC2 watersheds, this script may be helpful. 

The necessary data for producing the graph network are:
 * HUC8, HUC10, and HUC12 shapefiles from WBD
 * NHDWaterbody and NHDFlowlines shapefiles from NHDPlusV2
 * PlusFlow.csv from NHDPlusV2 (contains the TOCOMID and FROMCOMID connections)

If the user has the correct data (i.e., they have the NHDPlusV2 and WBD data for the area of interest), the following framework can be followed: 

 * Ensure all datasets use the same EPSG code (see lines 45-55 of `build_base_dataframes.py`)
 * Remove swamps/marshes from the waterbody dataset (if desired; see lines 62-63 of `build_base_dataframes.py`)
 * Add HUC code attributes to the WBD dataset (see lines 70-71 of `build_base_dataframes.py`; this is done through the function `add_huc_col_to_hucs`)
 * Add HUC code attributes to the waterbody and river datasets (lines 23-31 of file `add_hucs_to_lakes_rivers.py`; this is done through the functions `add_HUC8`, `add_HUC10`, and `add_HUC12`)
 * Identify the river segments that intersect each waterbody (and the waterbodies that intersect each river segment). This is done through the function `add_lake_river_node_column` (see line 12 of `add_river_lake_nodes.py`)
 * Aggregate the graph (if desired). This reduces the number of nodes in the graph, but maintains connectivity to HUC12s and upstream waterbodies. This is done through the function `aggregate_river_nodes` (see line 18 of `aggregated_graph`)

The result of this framework is an edge list (list of COMID sources and destinations). For river segments alone, this list is supplied by NHDPlusV2, but the above process returns a new list composed of lake and river COMIDs. In addition, the aggregation process returns a reduced list of COMIDs. These lists can be used very easily to construct graphs in `NetworkX` in Python. `HydroGraphs` also provides functions for producing the graph directly from the edge list/tofrom list, and it provides functions for getting the upstream and downstream graphs from a given COMID. 

## Correct Data Format

For the above code to run correctly, certain formatting must be followed. We have found that, depending on where the NHDPlusV2 data is retrieved from, slightly different naming conventions are used. For example, sometimes the COMID column of a data table is called "COMID" and sometimes it is called "ComID". For the above scripts to run accurately, all COMID columns must be named "COMID." In addition, all HUC units must be of type `integer`, where as they are sometimes type `string`. Lastly, HUC columns (i.e., HUC8, HUC10, HUC12) must be lowercase in the column name (i.e., columns must be named "huc8", "huc10", and "huc12"). 