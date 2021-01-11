# -*- coding: utf-8 -*-
"""
read_sampling_time_increment_varying_training_size.py
The purpose of this function is to vary the training size and time increment to 
make beautiful plots that show convergence to a certain extent

Created on: 06/07/2019

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
    
### FUNCTION TO EXTRACT TIME INCREMENT NAMES
def extract_time_size_dirname( directory ):
    '''
    This function extract information for a directory with time size variations
    INPUTS:
        directory: [str]
            name of the directory
    OUTPUTS:
        extracted_name: [dict]
            extracted details
    '''
    ## SPLITTING NAME
    split_name = directory.split('_')
    ## DEFINING DICTIONARY
    extracted_name = {
            'training_size': split_name[0],
            'test_size': split_name[1],
            }
    
    
    return extracted_name

######################################################
### CLASS FUNCTION TO GET SAMPLING TIME INCREMENTS ###
######################################################    
class read_sampling_time_increments_with_varying_training_sizes:
    '''
    This function extracts details for sampling time with increments
    INPUTS:
        path_sim_dir: [str]
            path to simulation directory
        image_file_path: [str, default = None]
            path to image file path
        results_pickle_file: [str, default = 'model.results']
            results pickle file name
        print_csv: [logical, default=False]
            True if you want output csvs
        save_fig: [logical, default=False]
            True if you want output figures and saved figures
        want_fig: [logical, default=False]
            True if you want output of learning curve, etc. for each system
    OUTPUTS:
        
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
        self.sim_directory_paths, self.sim_directory_basename, self.sim_directory_extracted_names = \
                        read_combined_name_directories(path = self.path_sim_dir, 
                                                       extraction_func = extract_time_size_dirname)
        
        ## EXTRACTING RMSES
        self.extract_rmse()
        
        return
    
    ### FUNCTION TO EXTRACT INFORMATION
    def extract_rmse(self):
        '''
        The purpose of this function is to extract rmse details from post-training.
        INPUTS:
            self: [obj]
                class object
        OUTPUTS:
            self.sim_rmse_storage: [list]
                list of RMSEs
            self.sim_perc_storage: [list]
                list of percentages
        '''
        ## CREATING EMPTY ARRAY TO STORE
        self.sim_rmse_storage = []
        self.sim_perc_storage =[]
    
        ## LOOPING THROUGH EACH SIMULATION DIRECTORY PATH
        for current_path_sim_dir in self.sim_directory_paths:
            
            ## EXTRACTING DIRECTORY NAMES
            directory_paths, directory_basename, directory_extracted_names = read_combined_name_directories(path = current_path_sim_dir)
    
            ## DEFINING EMPTY LIST
            rmse_storage, split_time_storage = [], []
            
            ## PRINTING
            print("Working on path: %s"%(current_path_sim_dir) )
            
            ## LOOPING THROUGH DIRECTORY INSTANCES
            for directory_instance in range(len(directory_extracted_names)):
                
                ### LOOPING THROUGH DIRECTORY INSTANCES
                current_directory_path = directory_paths[directory_instance]
                
                ## FINDING DIRECTORY NAME
                current_directory_name = directory_basename[directory_instance]
                
                ## PRINTING
                print("--> Working on directory: %s"%(current_directory_name) )
                
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
                mass_frac_data, \
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
                
                ## CLOSING ALL PLOTS
                if self.want_fig is True:
                    plt.close('all')
                
                ## STORING RMSE AND SPLITTING TIME
                rmse_storage.append(rmse)
                split_time_storage.append(representation_inputs['perc'])
            
            ## STORING
            self.sim_rmse_storage.append(rmse_storage)
            self.sim_perc_storage.append(split_time_storage)

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
    # r"/Volumes/akchew/scratch/3d_cnn_project/simulations"
    
    ## DEFAULT DIRECTORIES
    combined_database_path = r"R:\scratch\3d_cnn_project\combined_data_set"
    class_file_path =r"R:\scratch\3d_cnn_project\database\Experimental_Data\solvent_effects_regression_data.csv"
    ## DEFINING IMAGE FILE PATH
    image_file_path = r"R:\scratch\3d_cnn_project\images"
    ## DEFINING SIM PATH
    sim_path = r"R:\temp"
    # r"R:\scratch\3d_cnn_project\simulations"
    
    ## DEFINING MAIN DIRECTORY
    main_dir = r"190706-solvent_net_sample_increment_varying_training_size"
    # r"190618-solvent_net_sample_increment_varying_training_size"
    
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
    read_sampling_times = read_sampling_time_increments_with_varying_training_sizes( **inputs )
    
    #%%

    #%%
    
    ## IMPORTING FUNCTIONS
    from core.plotting_scripts import get_cmap
    
    ### FUNCTION TO MAKE A PLOT
    def plot_sampling_time_increments( read_sampling_times, 
                                       figsize,
                                       amount_ns_per_partition = 10,
                                       ):
        '''
        The purpose of this function is to plot RMSE vs. amount of training time
        INPUTS:
            read_sampling_times: [object]
                class object that contains all sampling time information
            amount_ns_per_partition: [float, default=10 ns]
                amount of nanoseconds per partition
        OUTPUTS:
            fig, ax: figure and axis for the plot
        '''
        ## DEFINING FIGURE
        fig = plt.figure() 
        ax = fig.add_subplot(111)
        
        ## SETTING AXIS LABELS
        ax.set_xlabel("Amount of simulation time per partition (ns)")
        ax.set_ylabel("RMSE")
        
        ## FINDING CMAP
        cmap = get_cmap(  len(read_sampling_times.sim_directory_extracted_names ) )
        
        ## LOOPING THROUGH EACH
        for idx, extract_names in enumerate(read_sampling_times.sim_directory_extracted_names):
            ## DEFINING X AND Y
            x = np.array(read_sampling_times.sim_perc_storage[idx])*amount_ns_per_partition
            y = read_sampling_times.sim_rmse_storage[idx]
            ## DEFINING COLOR
            current_color = cmap(idx)
            ## DEFINING LABEL
            label = 'num_training: %s' %( extract_names['training_size'] )
            ## PLOTTING
            # ax.scatter(x, y, color = current_color, linewidth=2, label = label )
            ax.plot(x, y, '.-', color = current_color, linewidth=2, label = label )
        
        ## ADDING Y LINE AT 0.10
        ax.axhline( y = 0.10, linestyle = '--', color = 'k', linewidth=2)
        
        ## ADDING LEGEND
        ax.legend()
        
        ## SHOW TIGHT LAYOUT
        fig.tight_layout()
        
        return fig, ax
    
    ## PLOTTING SAMPLING TIME INCREMENTS
    fig, ax = plot_sampling_time_increments( read_sampling_times = read_sampling_times, 
                                             figsize = (8,8))
    
    
    
    