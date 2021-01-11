#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
read_sampling_time_increments.py
The purpose of this script is to read sampling time data. We assume that sampling 
time was already run. Now, we want to extract the test RMSE across time increments. 


Created on: 05/27/2019

Author(s):
    - Alex K. Chew (alexkchew@gmail.com)
"""

## IMPORTING OS
import os
import pandas as pd
import glob
import matplotlib.pyplot as plt
import numpy as np

## IMPORTING FUNCTIONS
from core.import_tools import read_file_as_line
## IMPORTING GLOBAL VARIABLES
from core.global_vars import SOLUTE_TO_TEMP_DICT, CNN_DICT, SAMPLING_DICT
## IMPORTING PATHS
from core.path import read_combined_name_directories, extract_combined_names_to_vars

#######################
### 3D CNN NETWORKS ###
#######################
## IMPORTING COMBINING ARRAYS
from combining_arrays import combine_instances
## IMPORTING TRAIN DEEP CNN
from train_deep_cnn import train_deep_cnn
## IMPORTING ANALYSIS TOOL
from analyze_deep_cnn import analyze_deep_cnn, metrics
## IMPORITNG EXTRACTION PROTOCOL
from read_extract_deep_cnn import run_analysis
    



#%%
## MAIN FUNCTION
if __name__ == "__main__":
    ## DEFINING SOLVENT LIST
    full_solvent_list = [ 'DIO', 'GVL', 'THF' ]# , 'dmso' 'GVL', 'THF' ] # , 'GVL', 'THF' 
    ## DEFINING MASS FRACTION DATA
    full_mass_frac_data = ['10', '25', '50', '75'] # , '25', '50', '75'
    ## DEFINING SOLUTE LIST
    full_solute_list = list(SOLUTE_TO_TEMP_DICT)
    
    ## DEFINING DICTIONARY FOR CNN DICT
    cnn_dict = CNN_DICT
    
    ## DEFINING SAMPLING DICTIONARY
    sampling_dict = SAMPLING_DICT
    
    ## DEFINING VERBOSITY
    verbose = True
    
    ## SAVING FIGURE ?
    save_fig = True
    
    ## PRINTING CSV?
    print_csv = False
    
    ####################################
    ### DEFINING DIRECTORY AND PATHS ###
    ####################################
    
    ## DEFAULT DIRECTORIES
    combined_database_path = r"/Volumes/akchew/scratch/3d_cnn_project/combined_data_set"
    class_file_path =r"/Volumes/akchew/scratch/3d_cnn_project/database/Experimental_Datasolvent_effects_regression_data.csv"
    
    ## DEFINING IMAGE FILE PATH
    image_file_path = r"/Volumes/akchew/scratch/3d_cnn_project/images"
    
    ## DEFINING SIM PATH
    sim_path = r"/Volumes/akchew/scratch/3d_cnn_project/simulations"
    
    ## DEFINING MAIN DIRECTORY
    main_dir = r"190524-solvent_net_sample_increment_time_training"
    
    ###########################################    
    ### DEFINING INPUT DIRECTORIES AND PATH ###
    ###########################################
    ## DEFINING RESULTS PICKLE FILE
    results_pickle_file = r"model.results" # DIO 
    
    ## DEFINING PATH TO SIM
    path_sim_dir = os.path.join(  sim_path, main_dir )
    
    ###################
    ### MAIN SCRIPT ###
    ###################
    ## EXTRACTING DIRECTORY NAMES
    directory_paths, directory_basename, directory_extracted_names = read_combined_name_directories(path = path_sim_dir)
    
    ## DEFINING EMPTY LIST
    rmse_storage, split_time_storage = [], []
    
    ## DEFINING DIRECTORY INSTANCES
    directory_instance = 0
    for directory_instance in range(len(directory_extracted_names)):
    
        ### LOOPING THROUGH DIRECTORY INSTANCES
        current_directory_path = directory_paths[directory_instance]
        
        ## FINDING DIRECTORY NAME
        current_directory_name = directory_basename[directory_instance]
        
        ## CURRENT DIRECTORY EXTRACTED
        current_directory_extracted = directory_extracted_names[directory_instance]
        
        ## EXTRACTING INFORMATION
        representation_type, \
        representation_inputs, \
        data_type, \
        cnn_type, \
        num_epochs, \
        solute_list, \
        solvent_list, \
        mass_frac_data, = extract_combined_names_to_vars(extracted_name = current_directory_extracted)
        
        ## DEFINING PATHS
        results_full_path = os.path.join( current_directory_path, results_pickle_file )
        
        ## DEFINING ANALYSIS
        analysis = pd.read_pickle( results_full_path  )[0]
        
        ## RUNNING ANALYSIS
        total_training_time, rmse = run_analysis( 
                                      analysis = analysis,
                                      results_pickle_file = current_directory_name,
                                      image_file_path = image_file_path,
                                      print_csv = print_csv,
                                      save_fig = save_fig,
                                     )
        
        ## STORING RMSE AND SPLITTING TIME
        rmse_storage.append(rmse)
        split_time_storage.append(representation_inputs['perc'])
    
    #%%
    
    ## PLOTTING RMSE AND SPLIT TIME
    fig_name = None
    fig_extension = 'png'
    fig_suffix = 'sampling_time_increments'
    fig_path = image_file_path
    ## DEFINING FIGURE
    fig = plt.figure() 
    ax = fig.add_subplot(111)
    
    ## SETTING AXIS LABELS
    ax.set_xlabel("Percentage data used")
    ax.set_ylabel("Test RMSE")
    
    ## PLOTTING
    ax.scatter(split_time_storage, rmse_storage, color = 'k', linewidth=2 )
    
    ## ADDING Y LINE AT 0.10
    ax.axhline( y = 0.10, linestyle = '--', color = 'r', linewidth=2)
    
    ## SHOW TIGHT LAYOUT
    fig.tight_layout() # ADDED TO FIT THE CURRENT LAYOUT
    
    ## STORING FIGURE
    if save_fig == True:
        if fig_name is None:
            fig_name = main_dir + '_' + fig_suffix + '.' + fig_extension
        print("Printing figure: %s"%(fig_name) )
        fig.savefig( os.path.join(fig_path, fig_name), 
                     format=fig_extension, 
                     dpi = 1200,    )
    
    
    
    
    
    
    