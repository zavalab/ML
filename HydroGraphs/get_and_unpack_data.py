import wget
import os
import zipfile
import py7zr
import shutil

# Download agricultural land data

agland_url = "https://data.nal.usda.gov/system/files/WI_ACPFfields2019.zip"
path = os.getcwd()
wget.download(agland_url, out = path)

# Unpack data and remove zip file

agland_path = path + "/graph_construction/agland"

with zipfile.ZipFile("WI_ACPFfields2019.zip", 'r') as zipObj:
    zipObj.extract("WisconsinFieldBoundaries2019.cpg", path = agland_path)
    zipObj.extract("WisconsinFieldBoundaries2019.dbf", path = agland_path)
    zipObj.extract("WisconsinFieldBoundaries2019.prj", path = agland_path)
    zipObj.extract("WisconsinFieldBoundaries2019.sbn", path = agland_path)
    zipObj.extract("WisconsinFieldBoundaries2019.sbx", path = agland_path)
    zipObj.extract("WisconsinFieldBoundaries2019.shp", path = agland_path)
    zipObj.extract("WisconsinFieldBoundaries2019.shp.xml", path = agland_path)
    zipObj.extract("WisconsinFieldBoundaries2019.shx", path = agland_path)    

os.remove(path + "/WI_ACPFfields2019.zip")

# Download HUC04 NHDPlusV2 data

watershed4_url = "https://edap-ow-data-commons.s3.amazonaws.com/NHDPlusV21/Data/NHDPlusGL/NHDPlusV21_GL_04_NHDSnapshot_08.7z"
wget.download(watershed4_url, out = path)

# Extract data from .7z file

watershed4_path = path + "/graph_construction/lakes_rivers/Watershed4/"
with py7zr.SevenZipFile("NHDPlusV21_GL_04_NHDSnapshot_08.7z", 'r') as z:
    z.extractall(path)

# Move data to proper folder, and then delete the unnecessary unpacked data and the .7z file

NHDPlus4_path = path + "/NHDPlusGL/NHDPlus04/NHDSnapshot/Hydrography/"

shutil.move(NHDPlus4_path + "NHDFlowline.shp", watershed4_path)
shutil.move(NHDPlus4_path + "NHDFlowline.shp.xml", watershed4_path)
shutil.move(NHDPlus4_path + "NHDFlowline.dbf", watershed4_path)
shutil.move(NHDPlus4_path + "NHDFlowline.prj", watershed4_path)
shutil.move(NHDPlus4_path + "NHDFlowline.shx", watershed4_path)
shutil.move(NHDPlus4_path + "NHDWaterbody.shp", watershed4_path)
shutil.move(NHDPlus4_path + "NHDWaterbody.dbf", watershed4_path)
shutil.move(NHDPlus4_path + "NHDWaterbody.prj", watershed4_path)
shutil.move(NHDPlus4_path + "NHDWaterbody.shx", watershed4_path)

shutil.rmtree(path + "/NHDPlusGL")
os.remove(path + "/NHDPlusV21_GL_04_NHDSnapshot_08.7z")

# Download HUC07 NHDPlusV2 data

watershed7_url = "https://edap-ow-data-commons.s3.amazonaws.com/NHDPlusV21/Data/NHDPlusMS/NHDPlus07/NHDPlusV21_MS_07_NHDSnapshot_08.7z"
wget.download(watershed7_url, out = path)

# Extract data from .7z file

watershed7_path = path + "/graph_construction/lakes_rivers/Watershed7/"
with py7zr.SevenZipFile("NHDPlusV21_MS_07_NHDSnapshot_08.7z", 'r') as z:
    z.extractall(path)
    
# Move data to proper folder, and then delete the unnecessary unpacked data and the .7z file
    
NHDPlus7_path = path + "/NHDPlusMS/NHDPlus07/NHDSnapshot/Hydrography/"

shutil.move(NHDPlus7_path + "NHDFlowline.shp", watershed7_path)
shutil.move(NHDPlus7_path + "NHDFlowline.dbf", watershed7_path)
shutil.move(NHDPlus7_path + "NHDFlowline.prj", watershed7_path)
shutil.move(NHDPlus7_path + "NHDFlowline.shx", watershed7_path)
shutil.move(NHDPlus7_path + "NHDWaterbody.shp", watershed7_path)
shutil.move(NHDPlus7_path + "NHDWaterbody.dbf", watershed7_path)
shutil.move(NHDPlus7_path + "NHDWaterbody.prj", watershed7_path)
shutil.move(NHDPlus7_path + "NHDWaterbody.shx", watershed7_path)

shutil.rmtree(path + "/NHDPlusMS")
os.remove(path + "/NHDPlusV21_MS_07_NHDSnapshot_08.7z")

# Download the NHDPlusV2 Attributes to get file that contains the to-from comid lists for HUC 04

watershed4_url_attributes = "https://edap-ow-data-commons.s3.amazonaws.com/NHDPlusV21/Data/NHDPlusGL/NHDPlusV21_GL_04_NHDPlusAttributes_14.7z"
wget.download(watershed4_url_attributes, out = path)

# Extract data from the .7z file

watershed4_path = path + "/graph_construction/lakes_rivers/Watershed4/"
with py7zr.SevenZipFile("NHDPlusV21_GL_04_NHDPlusAttributes_14.7z", 'r') as z:
    z.extractall(path)

# Move data to proper folder, and then delete the unnecessary unpacked data and the .7z file

NHDPlus4_attributes_path = path + "/NHDPlusGL/NHDPlus04/NHDPlusAttributes/"

shutil.move(NHDPlus4_attributes_path + "PlusFlow.dbf", watershed4_path)

shutil.rmtree(path + "/NHDPlusGL")
os.remove(path + "/NHDPlusV21_GL_04_NHDPlusAttributes_14.7z")

# Download the NHDPlusV2 Attributes to get file that contains the to-from comid lists for HUC 07

watershed7_url_attributes = "https://edap-ow-data-commons.s3.amazonaws.com/NHDPlusV21/Data/NHDPlusMS/NHDPlus07/NHDPlusV21_MS_07_NHDPlusAttributes_10.7z"
wget.download(watershed7_url_attributes, out = path)

# Extract data from the .7z file

watershed7_path = path + "/graph_construction/lakes_rivers/Watershed7/"
with py7zr.SevenZipFile("NHDPlusV21_MS_07_NHDPlusAttributes_10.7z", 'r') as z:
    z.extractall(path)
    
# Move data to proper folder, and then delete the unnecessary unpacked data and the .7z file
    
NHDPlus7_attributes_path = path + "/NHDPlusMS/NHDPlus07/NHDPlusAttributes/"

shutil.move(NHDPlus7_attributes_path + "PlusFlow.dbf", watershed7_path)

shutil.rmtree(path + "/NHDPlusMS")
os.remove(path + "/NHDPlusV21_MS_07_NHDPlusAttributes_10.7z")

# Download the Watershed Boundary Dataset for HUC04

wbd4_url = "https://prd-tnm.s3.amazonaws.com/StagedProducts/Hydrography/WBD/HU2/Shape/WBD_04_HU2_Shape.zip"
wget.download(wbd4_url, out = path)

# Unpack the dataset, move the necessary files, and remove the remaining files and zip file

watershed4_path = path + "/graph_construction/lakes_rivers/Watershed4/"

with zipfile.ZipFile("WBD_04_HU2_Shape.zip", 'r') as zipObj:
    zipObj.extractall(path = path)

shutil.move(path + "/Shape/WBDHU8.dbf", watershed4_path)
shutil.move(path + "/Shape/WBDHU8.prj", watershed4_path)
shutil.move(path + "/Shape/WBDHU8.shp", watershed4_path)
shutil.move(path + "/Shape/WBDHU8.shx", watershed4_path)
shutil.move(path + "/Shape/WBDHU10.dbf", watershed4_path)
shutil.move(path + "/Shape/WBDHU10.prj", watershed4_path)
shutil.move(path + "/Shape/WBDHU10.shp", watershed4_path)
shutil.move(path + "/Shape/WBDHU10.shx", watershed4_path)
shutil.move(path + "/Shape/WBDHU12.dbf", watershed4_path)
shutil.move(path + "/Shape/WBDHU12.prj", watershed4_path)
shutil.move(path + "/Shape/WBDHU12.shp", watershed4_path)
shutil.move(path + "/Shape/WBDHU12.shx", watershed4_path)
    
os.remove(path + "/WBD_04_HU2_Shape.zip")
os.remove(path + "/WBD_04_HU2_Shape.jpg")
os.remove(path + "/WBD_04_HU2_Shape.xml")
shutil.rmtree(path + "/Shape")

# Download the Watershed Boundary Dataset for HUC07

wbd7_url = "https://prd-tnm.s3.amazonaws.com/StagedProducts/Hydrography/WBD/HU2/Shape/WBD_07_HU2_Shape.zip"
wget.download(wbd7_url, out = path)

# Unpack the dataset, move the necessary files, and remove the remaining files and zip file

watershed7_path = path + "/graph_construction/lakes_rivers/Watershed7/"

with zipfile.ZipFile("WBD_07_HU2_Shape.zip", 'r') as zipObj:
    zipObj.extractall(path = path)

shutil.move(path + "/Shape/WBDHU8.dbf", watershed7_path)
shutil.move(path + "/Shape/WBDHU8.prj", watershed7_path)
shutil.move(path + "/Shape/WBDHU8.shp", watershed7_path)
shutil.move(path + "/Shape/WBDHU8.shx", watershed7_path)
shutil.move(path + "/Shape/WBDHU10.dbf", watershed7_path)
shutil.move(path + "/Shape/WBDHU10.prj", watershed7_path)
shutil.move(path + "/Shape/WBDHU10.shp", watershed7_path)
shutil.move(path + "/Shape/WBDHU10.shx", watershed7_path)
shutil.move(path + "/Shape/WBDHU12.dbf", watershed7_path)
shutil.move(path + "/Shape/WBDHU12.prj", watershed7_path)
shutil.move(path + "/Shape/WBDHU12.shp", watershed7_path)
shutil.move(path + "/Shape/WBDHU12.shx", watershed7_path)

os.remove(path + "/WBD_07_HU2_Shape.zip")
os.remove(path + "/WBD_07_HU2_Shape.jpg")
os.remove(path + "/WBD_07_HU2_Shape.xml")
shutil.rmtree(path + "/Shape")

# Unpack the zipfile with WILakes.df and WIRivers.df

graph_construction_path = path + "/graph_construction"

with zipfile.ZipFile(graph_construction_path + "/WI_lakes_rivers.zip", "r") as zipObj:
    zipObj.extract("WILakes.df", path = graph_construction_path)
    zipObj.extract("WIRivers.df", path = graph_construction_path)

# Unpack the zipfile with the Hydro Waterbodies dataset for Wisconsin

Hydro_path = path + "/DNR_data/Hydro_Waterbodies"

with zipfile.ZipFile(Hydro_path + "/24k_Hydro_Waterbodies_(Open_Water).zip", "r") as zipObj:
    zipObj.extract("24k_Hydro_Waterbodies_(Open_Water).xml", path = Hydro_path)
    zipObj.extract("24k_Hydro_Waterbodies_(Open_Water).cpg", path = Hydro_path)    
    zipObj.extract("24k_Hydro_Waterbodies_(Open_Water).dbf", path = Hydro_path)    
    zipObj.extract("24k_Hydro_Waterbodies_(Open_Water).prj", path = Hydro_path)    
    zipObj.extract("24k_Hydro_Waterbodies_(Open_Water).shp", path = Hydro_path)    
    zipObj.extract("24k_Hydro_Waterbodies_(Open_Water).shx", path = Hydro_path)    
