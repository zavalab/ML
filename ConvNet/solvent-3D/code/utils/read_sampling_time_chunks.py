# -*- coding: utf-8 -*-
"""
read_sampling_time_chunks.py
The purpose of this script is to read sampling time chunks. The whole goal 
is to sample across the trajectory with a fixed range to see when is the earliest 
time we could make favorable predictions. 

Created on: 05/31/2019

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
## IMPORTING NOMENCLATURE
from core.nomenclature import read_combined_name, extract_representation_inputs

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
from read_sampling_time_increments import extract_combined_names_to_vars, read_combined_name_directories

### CLASS FUNCTION TO RUN SAMPLING TIME CHUNKS
class read_sampling_time_chunks:
    '''
    The purpose of this function is to extract sampling time chunk details
    '''
    ## INITIALIZING
    def __init__(self,
                 path_sim_dir,
                 results_pickle_file = r"model.results",
                 image_file_path = None,
                 print_csv = False,
                 save_fig = False,
                 want_fig = False,
                 ):
        ## STORING INPUTS
        self.path_sim_dir = path_sim_dir
        self.image_file_path = image_file_path
        self.results_pickle_file = results_pickle_file
        self.print_csv = print_csv
        self.save_fig = save_fig
        self.want_fig = want_fig

        ## EXTRACTING DIRECTORY NAMES
        self.directory_paths, self.directory_basename, self.directory_extracted_names = \
                        read_combined_name_directories(path = self.path_sim_dir, 
                                                       )
        ## RUNNING EXTRACTION PROCESS
        self.extract_rmse()
                        
    ### FUNCTION TO EXTRACT INFORMATION
    def extract_rmse(self):
        '''
        This extracts rmse details
        INPUTS:
            self: [obj]
                class object
        OUTPUTS:
            
        '''
        ## DEFINING EMPTY LIST
        self.rmse_storage, self.split_time_storage = [], []
        
        ## DEFINING DIRECTORY INSTANCES
        for directory_instance in range(len(self.directory_extracted_names)):
            
            ### LOOPING THROUGH DIRECTORY INSTANCES
            current_directory_path = self.directory_paths[directory_instance]
            
            ## FINDING DIRECTORY NAME
            current_directory_name = self.directory_basename[directory_instance]
            
            ## PRINTING
            print("--> Working on directory: %s"%(current_directory_name) )
            ## CURRENT DIRECTORY EXTRACTED
            current_directory_extracted = self.directory_extracted_names[directory_instance]
            
            ## EXTRACTING INFORMATION
            representation_type, \
            representation_inputs, \
            sampling_dict, \
            data_type, \
            cnn_type, \
            num_epochs, \
            solute_list, \
            solvent_list, \
            mass_frac_data,\
            want_descriptors = extract_combined_names_to_vars(extracted_name = current_directory_extracted)
            
            ## DEFINING PATHS
            results_full_path = os.path.join( current_directory_path, self.results_pickle_file )
            
            ## DEFINING ANALYSIS
            analysis = pd.read_pickle( results_full_path  )[0]
            
            ## RUNNING ANALYSIS
            total_training_time, rmse = run_analysis( 
                                          analysis = analysis,
                                          results_pickle_file = current_directory_name,
                                          image_file_path = self.image_file_path,
                                          print_csv = self.print_csv,
                                          save_fig = self.save_fig,
                                          want_fig = self.want_fig,
                                         )
            
            ## STORING RMSE AND SPLITTING TIME
            self.rmse_storage.append(rmse)
            self.split_time_storage.append([representation_inputs['initial_frame'],representation_inputs['last_frame']])
#%%
## MAIN FUNCTION
if __name__ == "__main__":

    
    ####################################
    ### DEFINING DIRECTORY AND PATHS ###
    ####################################

    
    ## DEFAULT DIRECTORIES
    combined_database_path = r"R:\scratch\3d_cnn_project\combined_data_set"
    class_file_path =r"R:\scratch\3d_cnn_project\database\Experimental_Data\solvent_effects_regression_data.csv"
    
    ## DEFINING IMAGE FILE PATH
    image_file_path = r"R:\scratch\3d_cnn_project\images"
    
    ## DEFINING SIM PATH
    sim_path =  r"R:\scratch\3d_cnn_project\simulations"
    
    ## DEFINING MAIN DIRECTORY
    main_dir = r"190618-solvent_net_sampling_chunks"
    main_dir = r"190702-solvent_net_sampling_chunks"
    main_dir = r"190708-solvent_net_sampling_chunks_20ns"
    ###########################################    
    ### DEFINING INPUT DIRECTORIES AND PATH ###
    ###########################################
    ## DEFINING RESULTS PICKLE FILE
    results_pickle_file = r"model.results" # DIO 
    
    ## DEFINING PATH TO SIM
    path_sim_dir = os.path.join(  sim_path, main_dir )
    
    ## DEFINING INPUTS
    inputs={
            'path_sim_dir': path_sim_dir,
            'results_pickle_file': results_pickle_file,
            'image_file_path': None,
            'print_csv': False,
            'save_fig' : False,
            'want_fig' : False,
            }
    
    ## RUNNING ANALYSIS
    sampling_time_chunks = read_sampling_time_chunks( **inputs )
    
    
    #%%
    ###################
    ### MAIN SCRIPT ###
    ###################
    ## EXTRACTING DIRECTORY NAMES
    directory_paths, directory_basename, directory_extracted_names = read_combined_name_directories(path = path_sim_dir)
    
    #%%
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
        sampling_dict, \
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
        split_time_storage.append([representation_inputs['initial_frame'],representation_inputs['last_frame']])
        
    #%%
    
    ## DEFINING INITIAL TIMES
    initial_times = [ each_time[0] for each_time in sampling_time_chunks.split_time_storage]
    
    
    ## PLOTTING RMSE AND SPLIT TIME
    fig_name = None
    fig_extension = 'png'
    fig_suffix = 'sampling_time_chunks'
    fig_path = image_file_path
    save_fig = False
    ## DEFINING FIGURE
    fig = plt.figure() 
    ax = fig.add_subplot(111)
    
    ## SETTING AXIS LABELS
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Test RMSE")
    
    ## DEFINING TICKS
    x_ticks = (1000, 11000, 1000 ) # 0.5
    y_ticks = (0, 1, 0.1 ) # 0.5
    
    ## DEFINING XY LIMITS
    x_lims = (1000, 11000)
    y_lims = (0, 0.5)
    
    ## SETTING X TICKS AND Y TICKS
    ax.set_xticks(np.arange(x_ticks[0], x_ticks[1] + x_ticks[2], x_ticks[2]))
    ax.set_yticks(np.arange(y_ticks[0], y_ticks[1] + y_ticks[2], y_ticks[2]))
    
    ax.set_xlim([x_lims[0], x_lims[1]] )
    ax.set_ylim([y_lims[0], y_lims[1]])
    
    ## LOOPING OVER EACH
    for idx, each_time in enumerate(sampling_time_chunks.split_time_storage):
        ## PLOTTING A HORIZONTAL LINE
        ax.hlines( y = sampling_time_chunks.rmse_storage[idx], xmin = each_time[0], xmax = each_time[1], colors = 'k', linestyles = 'solid'   )
    
    ## PLOTTING
    # ax.scatter(initial_times, rmse_storage, color = 'k', linewidth=2 )
    
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
    
    