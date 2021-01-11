# -*- coding: utf-8 -*-
"""
read_cross_validation.py
The purpose of this script is to read the cross validation results. We assume 
that you have already completed the cross-validation calculations. 

Created on: 05/16/2019

Author(s):
    - Alex K. Chew (alexkchew@gmail.com)
        
    
FUNCTIONS:
    find_ordered_list_index: reorders lists
"""

## IMPORTING OS
import os
import pandas as pd
import glob
import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys

## IMPORTING FUNCTIONS
from core.import_tools import read_file_as_line
## IMPORTING GLOBAL VARIABLES
from core.global_vars import SOLUTE_TO_TEMP_DICT, CNN_DICT, SAMPLING_DICT, INPUTS_FOR_DESCRIPTOR_FXN, ORDER_OF_REACTANTS
## IMPORTING NOMENCLATURE
from core.nomenclature import read_combined_name

## IMPORTING PATHS
from core.path import read_combined_name_directories, extract_combined_names_to_vars

## IMPORTING PATH TO CHECK PATHS
from core.check_tools import check_path_to_server, check_multiple_paths

#######################
### 3D CNN NETWORKS ###
#######################
## IMPORTING COMBINING ARRAYS
from combining_arrays import combine_instances
## IMPORTING TRAIN DEEP CNN
from train_deep_cnn import train_deep_cnn
## IMPORTING ANALYSIS TOOL
from analyze_deep_cnn import analyze_deep_cnn, metrics

## IMPORTING PUBLISHABLE FIGURES
from read_extract_deep_cnn import plot_parity_publication_single_solvent_system

from extraction_scripts import load_pickle,load_pickle_general

## MD DESCRIPTORS
from deep_cnn_md_descriptors import get_descriptor_inputs

    
## FUNCTION TO FIND REORDERED LIST
def find_ordered_list_index(list_to_order,
                            ordered_list = ORDER_OF_REACTANTS ):
    '''
    The purpose of this function is to find the ordered list index. If the 
    list you want to order does not have the ordered list information, it will 
    append to the end of the list.
    INPUTS:
        list_to_order: [list]
            list you want to order
        ordered_list: [list]
            list that has the order you want
    OUTPUTS:
        reordered_list_index: [np.array]
            index list with the same shape as the list to order with the desired order
    '''
    ## FINDING TOTAL LIST
    total_index_list = np.arange(len(list_to_order))
    
    ## DEFINING REORDERD LIST
    reordered_list_index = []
    
    ## RE-ORDERING AS NECESSARY
    for each_index, var_name in enumerate(ordered_list):
        ## SEEING IF WITHIN THE LIST
        if var_name in list_to_order:
            ## FINDING INDEX
            index_of_var = list_to_order.index(var_name)
            ## STORING TO NEW LIST
            reordered_list_index.append(index_of_var)
    
    ## GENERATING NUMPY ARRAY
    reordered_list_index = np.array(reordered_list_index)
    
    ## SEEING IF THERE ARE INDICES NOT IN LIST
    is_within_index = np.nonzero(np.isin(total_index_list, reordered_list_index, invert = True))[0]
    
    ## APPENDING
    reordered_list_index = np.append(reordered_list_index, is_within_index).astype(int)
    return reordered_list_index


##################################################
### CLASS FUNCTION TO ANALYZE CROSS VALIDATION ###
##################################################
class analyze_cross_validation:
    '''
    The purpose of this function is to analyze the results from cross validation. 
    INPUTS:
        main_dir: [str]
            main directory within simulation path that you are observing
        combined_database_path: [str]
            combined dataset path, used to extract instances
        class_file_path: [str]
            class file path to regression data, used for instances
        image_file_path: [str]
            image file path
        database_path: [str]
            path to the database
        sim_path: [str]
            simulation path
        results_pickle_file: [str, default='model.results']
            pickle file for the model
        cross_validation_file: [str, default='cross_valid.txt']
            cross validation file containing all information
        verbose: [logical, default=False]
            True if you want output to be verbose
        num_cross_validation_folds: [int]
            number of cross validation folds. Default is 5.
    '''
    ## INITIALIZING
    def __init__(self,
                 main_dir,
                 combined_database_path,
                 class_file_path,
                 image_file_path,
                 database_path,
                 sim_path,
                 results_pickle_file = r"model.results",
                 cross_validation_file = r"cross_valid.txt",
                 verbose = False,
                 num_cross_validation_folds = 5,
                 ):
        ## STORING INPUT VARIABLES
        self.main_dir = main_dir
        self.combined_database_path = combined_database_path
        self.class_file_path = class_file_path
        self.image_file_path = image_file_path
        self.sim_path = sim_path
        self.database_path = database_path
        self.results_pickle_file = results_pickle_file
        self.cross_validation_file = cross_validation_file
        self.verbose = verbose
        self.num_cross_validation_folds = num_cross_validation_folds
        
        ## FINDING PATHS
        self.find_paths()
        
        ## DEFINING GLOBAL VARIABLES
        self.cnn_dict = CNN_DICT
        
        ########################################
        ### EXTRACTING CROSS VALIDATION DATA ###
        ########################################
        self.extract_cross_validation_file()
        
        ## EXTRACTING DIRECTORY NAMES
        self.cross_valid_directories, self.cross_valid_directories_dirname, self.cross_valid_directories_extracted = \
                                            read_combined_name_directories(path = self.path_sim_dir)
        
        ## RUNNING CROSS VALIDATION
        self.extract_validation_from_cnns()
        
        return

    ## FUNCTION TO EXTRACT DEEP CROSS VALIDATION INFORMATION
    def extract_validation_from_cnns(self):
        '''
        The purpose of this function is to loop through each directory and extract 
        cross validation details from the CNNs. 
        
        '''
        ## CREATING STORAGE LIST
        self.cross_validation_storage = []
        
        ## DEFINING RANGE OF DIRECTORIES
        directory_range = range(len(self.cross_valid_directories_extracted))
        # directory_range = [0]
        
        ## LOOPING THROUGH DIRECTORIES
        for directory_instance in directory_range:
            
            ## DEFINING CURRENT DIRECTORY
            current_directory = self.cross_valid_directories[directory_instance]
            
            ## GETTING EXTRACTED VALUES
            current_directory_extracted = self.cross_valid_directories_extracted[directory_instance]
            
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
            want_descriptors= extract_combined_names_to_vars(extracted_name = current_directory_extracted)
            
            ## LOADING ALL INSTANCE DATA
            if directory_instance == 0:
                ## DEFINING DATABASE PATH
                self.database_path = os.path.join( self.database_path, data_type)  # None # Since None, we will find them!        
                self.instances = combine_instances(
                                         solute_list = self.full_variable_list['solute'],
                                         representation_type = representation_type,
                                         representation_inputs = representation_inputs,
                                         solvent_list = self.full_variable_list['cosolvent'], 
                                         mass_frac_data = self.full_variable_list['mass_frac'], 
                                         verbose = self.verbose,
                                         database_path = self.database_path,
                                         class_file_path = self.class_file_path,
                                         combined_database_path = self.combined_database_path,
                                         data_type = data_type,
                                         )
            
            ## FINDING WHICH SOLUTES ARE NOT WITHIN THE SET
            if self.cross_validation_name == "solute":
                test_set_variables = list(set(solute_list) ^ set(self.full_variable_list['solute']))[0]
            elif self.cross_validation_name == "cosolvent":
                test_set_variables = list(set(solvent_list) ^ set(self.full_variable_list['cosolvent']))[0]
            elif self.cross_validation_name == "mass_frac":
                test_set_variables = list(set(mass_frac_data) ^ set(self.full_variable_list['mass_frac']))[0]
            print("---------------------------------------------")
            print("Cross validation name: %s -- %s"%(self.cross_validation_name, test_set_variables) )
            print("---------------------------------------------")
            ### TRAINING CNN (WITH NEW TEST SET)
            deep_cnn = train_deep_cnn(
                             instances = self.instances,
                             sampling_dict = sampling_dict,
                             cnn_type = cnn_type,
                             cnn_dict = self.cnn_dict,
                             retrain=False,
                             output_path = current_directory,
                             verbose = self.verbose,
                             want_training_test_pickle = False,
                             want_basic_name = True,
                             want_descriptors = want_descriptors,
                             num_cross_validation_folds = self.num_cross_validation_folds,
                             )
            
            
            ###################################
            ### EXTRACTING CURRENT ANALYSIS ###
            ###################################
            ## DEFINING FULL PATH
            path_results = os.path.join( self.path_sim_dir, current_directory, self.results_pickle_file )
            
            ## READING ANALYSIS FILE
            analysis_training = pd.read_pickle( path_results )[0]
            
            ## STORING
            # self.deep_cnn = deep_cnn
            # self.path_results = path_results
            # self.analysis_training = analysis_training
            
            ############################
            ### RUNNING NEW ANALYSIS ###
            ############################
            ### RE-RUNNING ANALYZE DEEP CNN
            analysis_new_test_set = analyze_deep_cnn( 
                                                        instances = self.instances,
                                                        deep_cnn = deep_cnn, 
                                                    )
            ## FINDING RMSE OF TEST SET
            test_set_dataframe = analysis_new_test_set.dataframe[analysis_new_test_set.dataframe[self.cross_validation_name] == test_set_variables]
            
            ## GETTING PRED VALUES
            y_true = np.array(test_set_dataframe['y_true'])
            y_pred = np.array(test_set_dataframe['y_pred'])
            
            ## COMPUTING RMSE
            test_rmse = metrics( y_fit = y_pred,y_act = y_true )[1]
            
            ## DEFINING CROSS VALIDATION INPUT
            cross_validation_training_info = {
                    'training_slope': analysis_training.accuracy_dict['slope'],
                    'training_rmse':  analysis_training.accuracy_dict['rmse'],
                    'cross_validation_name': self.cross_validation_name,
                    'test_set_variables': test_set_variables,
                    'test_rmse': test_rmse,
                    'alpha': 0.1, # 0.1
                    }
            
            ## STORING TRAINING DETAILS
            self.cross_validation_storage.append(
                    {
                            'current_directory': current_directory,
                            'current_directory_extracted': current_directory_extracted,
                            'dataframe': analysis_new_test_set.dataframe,
                            'cross_validation_training_info': cross_validation_training_info,
                            }
                    )
        return
    
    ## FUNCTION TO FIND PATHS
    def find_paths(self):
        ''' This function finds paths based on inputs'''
        ## DEFINING PATH TO SIM
        self.path_sim_dir = os.path.join( self.sim_path, self.main_dir )
        self.path_cross_validation_file = os.path.join(  self.path_sim_dir,  self.cross_validation_file )
        return

    ## FUNCTION TO FIND ALL POSSIBLE DETAILS
    def extract_cross_validation_file(self):
        '''
        The purpose of this function is to extract cross validation file to get 
        full solute, etc. list
        
        === Cross validation file example ===
        
            ['cross_valid.txt',
             '',
             '--- VARIABLES_NOT_VARIED_START ---',
             'mass_frac 10,25,50,75',
             'solute CEL,ETBE,FRU,LGA,PDO,XYL,tBuOH',
             '--- VARIABLES_NOT_VARIED_END ---',
             '',
             'CROSS_VALIDATION_NAME: cosolvent',
             'CROSS_VALIDATION_TOTAL: 3',
             '--- CROSS_VALIDATION_START ---',
             'GVL,THF',
             'DIO,THF',
             'DIO,GVL',
             '--- CROSS_VALIDATION_END ---']
        '''
        ## READING CROSS VALIDATION FILE
        self.cross_validation_data = read_file_as_line(file_path = self.path_cross_validation_file,
                                                       want_clean = True)
        
        ## FINDING CROSS VALIDATION NAME
        self.cross_validation_name = [ each_line.split()[1] for each_line in self.cross_validation_data 
                                             if 'CROSS_VALIDATION_NAME' in each_line ][0]
        
        ## FINDING INDEX OF VARIABLES NOT VARIED
        index_not_varied_initial = self.cross_validation_data.index('--- VARIABLES_NOT_VARIED_START ---')
        index_not_varied_final = self.cross_validation_data.index('--- VARIABLES_NOT_VARIED_END ---')
        
        ## LOOPING THROOUGH INDICES
        list_not_varied = self.cross_validation_data[index_not_varied_initial+1:index_not_varied_final]
        
        ## DEFINING EMPTY DICTIONARY OF FULL LIST
        self.full_variable_list = {}
        
        ## LOOPING THROUGH LIST AND STORING
        for each_list in list_not_varied:
            ## SPLITTING BASED ON SPACE
            current_list_split = each_list.split()
            ## STORING VARIABLE
            self.full_variable_list[current_list_split[0]] = current_list_split[1].split(',')
        
        ## FINDING INDICES OF VARIABLES THAT ARE VARIED
        index_varied_initial = self.cross_validation_data.index('--- CROSS_VALIDATION_START ---')
        index_varied_final = self.cross_validation_data.index('--- CROSS_VALIDATION_END ---')
        
        ## DEFINING LIST
        list_varied = self.cross_validation_data[index_varied_initial+1:index_varied_final]
        
        ## COMBINING ALL LIST
        unique_list = np.unique(np.array([ each_list.split(',') for each_list in list_varied ]))
        self.full_variable_list[self.cross_validation_name] = unique_list.tolist()
        
        return

    
### FUNCTION TO PLOT ALL CNNS
def plot_all_cross_validations(cross_validation_storage,
                               parity_plot_inputs,
                               want_combined_plot = False,
                               want_reorder = True,
                               ):
    '''
    The purpose of this function is to plot all cross validations.
    INPUTS:
        cross_validation_storage: [list]
            cross validation storage containing all cross validation information
        parity_plot_inputs: [dict]
            dictionary for inputs to parity plot 'plot_parity_publication_single_solvent_system'
        want_combined_plot: [logical, default = False]
            True if you want combined plot (single plot)
        want_reorder: [logical,default=True]
            True if you want to reorder the cross validation results
    '''
    ## SEEING IF YOU WANT COMBINED PATH
    if want_combined_plot is True:
        fig = None
        ax = None
        stored_save_fig = parity_plot_inputs['save_fig']
        
        ## DEFINING STORAGE
        cross_validation_training_info_stored = {'last_one': False,
                                                 'data': []}
    
    ## REORDERING CROSS VALIDATION DATA
    if want_reorder is True:
        ## DEFINING CROSS VALIDATION STORAGE INFO
        test_set_variable_name = [ each_list['cross_validation_training_info']['test_set_variables']  \
                                    for each_list in cross_validation_storage ]
        
        ## GETTING REORDERED IS INDEX
        reordered_list_index = find_ordered_list_index(list_to_order = test_set_variable_name,
                                                       ordered_list = ORDER_OF_REACTANTS )
        
        
        ## REDEFINING CROSS VALIDATION STORAGE
        cross_validation_storage = [ cross_validation_storage[each_index] for each_index in reordered_list_index]
    
    ## LOOPING THROUGH EACH LIST
    for idx, each_list in enumerate(cross_validation_storage):
        ## DEFINING DATAFRAME
        dataframe = each_list['dataframe']
        ## DEFINING CROSS VALIDATION TRAINING INFORMATION
        cross_validation_training_info = each_list['cross_validation_training_info']
        
        if want_combined_plot is True:
            ## TURNING ALL TRAINING DATA OFF
            cross_validation_training_info['alpha'] = 0
            ## STORING DATA
            cross_validation_training_info_stored['data'].append(cross_validation_training_info)
            if idx != len(cross_validation_storage) - 1:
                parity_plot_inputs['save_fig'] = False
            else:
                cross_validation_training_info_stored['last_one'] = True
                parity_plot_inputs['save_fig'] = stored_save_fig
            
                
        ## DEFINING FIGURE NAME
        ## GENERATING A PLOT
        if want_combined_plot is False:
            fig, ax = plot_parity_publication_single_solvent_system( 
                                                           dataframe = dataframe,
                                                           cross_validation_training_info = cross_validation_training_info,
                                                           **parity_plot_inputs
                                                           )
        else:
            fig, ax = plot_parity_publication_single_solvent_system( 
                                                           dataframe = dataframe,
                                                           cross_validation_training_info = cross_validation_training_info,
                                                           cross_validation_training_info_stored = cross_validation_training_info_stored,
                                                           fig = fig,
                                                           ax = ax,
                                                           **parity_plot_inputs
                                                           )
            
    return fig, ax

### FUNCTION TO GET ALL TEST SET INFORMATION
def get_test_set_df_from_cross_valid(cross_valid_results,
                                     NN_inputs = None):
    '''
    The purpose of this function is to get test dataframe for each cross validation storage. 
    INPUTS:
        cross_valid_results: [object]
            output from 'analyze_cross_validation' function
        NN_inputs: [dict]
            inputs for artificial neural network approach
                'cross_validation_name': 'solute' or 'cosolvent'
    OUTPUTS:
        test_set_df_full: [dataframe]
            dataframe with test set data ONLY
    '''
    ## STORING DATAFRAME
    test_set_df_storage = []
    # print(cross_valid_results)
    ## DEFINING STORAGE 
    if NN_inputs is None:
        storage_details = cross_valid_results.cross_validation_storage
    else:
        storage_details = cross_valid_results
        
    ## LOOPING
    for cross_valid_storage in storage_details:
        
        ## IF NO NEURAL NETWORK IS USED
        if NN_inputs is None:
            ## FINDING TEST SET INFORMATION
            cross_valid_training_info = cross_valid_storage['cross_validation_training_info']
            
            ## COLUMN NAME AND VALUE
            column_name = cross_valid_training_info['cross_validation_name']
            value = cross_valid_training_info['test_set_variables']
            
            ## DEFINING DATAFRAME
            dataframe = cross_valid_storage['dataframe']
        else:
            ## DEFINING COLUMN NAME
            column_name = NN_inputs['cross_validation_name']
            ## DEFINING VALUE
            value = cross_valid_storage
            ## DEFINING DATAFRAME
            dataframe = storage_details[cross_valid_storage]['predict_df'][0]
        
        ## FIXING IF TERT-BUTANOL NOMENCLATURE
        if value == 'tBuOH': ## FIXING
            value = 'TBA'
                    
        ## FINDING TEST SET DATAFRAME
        test_set_df = dataframe[dataframe[column_name] == value]
        ## APPENDING
        test_set_df_storage.append(test_set_df)
    
    ## CONCATENATING PDs
    test_set_df_full = pd.concat(test_set_df_storage)
    return test_set_df_full

### FUNCTION TO GET CUMULATIVE RMSE
def compute_cumulative_rmse( test_set_df_full ):
    '''
    The purpose of this function is to get the cumulative RMSE given the cross 
    validation results. This will go through all test set RMSE, then finds 
    the error. The error is then re-ordered. Finally, the RMSE is computed as a cumulative 
    function. 
    INPUTS:
        test_set_df_full: [dataframe]
            dataframe with test set data ONLY
    OUTPUTS:
        cumulative_RMSE: [np.array]
            cumulative root-mean-squared erorr
    '''
    ## GETTING PREDICTED AND TRUE VALUES
    predicted_and_true_values = np.array(test_set_df_full[['y_pred','y_true']])
    
    ## SUBTRACTING PRED AND TRUE VALUES
    difference_btwn_pred_and_true = (predicted_and_true_values[:,1]  - predicted_and_true_values[:,0])**2
    
    ## REORDER FROM SMALLEST TO LARGEST
    ordered_difference = np.sort(difference_btwn_pred_and_true) # [::-1]
    
    ## GETTING CUMULATIVE RMSE
    cumulative_diff = np.cumsum(ordered_difference)
    
    ## GETTING TOTAL INSTANCES
    total_instances_array = np.arange(1,len(cumulative_diff)+1)
    
    ## CUMULATIVE RMSE
    cumulative_RMSE = np.sqrt( cumulative_diff / total_instances_array )
    
    return cumulative_RMSE

### FUNCTION TO GET STATS FROM CROSS VALIDATION
def compute_stats_from_cross_valid(test_set_df_full, 
                                   desired_stats = ['slope', 'rmse', 'pearson_r'],
                                   y_pred_key = 'y_pred',
                                   y_true_key = 'y_true',
                                   ):
    '''
    The purpose of this script is to compute stats from cross validation.
    INPUTS:
        test_set_df_full: [dataframe]
            dataframe with test set data ONLY
        desired_stats: [list]
            list of stats names that you want
        y_pred_key: [str]
            prediction key
        y_true_key: [str]
            actual value key
    OUTPUTS:
        output_stats: [dict]
            dictionary of output stats
    '''
    ## GETTING PREDICTED VALUES AND EXP VALUE
    predicted_values = np.array(test_set_df_full[y_pred_key])
    actual_values = np.array(test_set_df_full[y_true_key])
    
    ## COMPUTING METRICS
    current_metrics = metrics(y_fit = predicted_values,
                              y_act = actual_values, 
                              want_dict = True )
    
    
    ## DEFINING DESIRED STATS
    desired_stats = ['slope', 'rmse', 'pearson_r']
    
    ## LOOPING TO GET DESIRED STATS
    output_stats =  {stat_name:current_metrics[stat_name] for stat_name in desired_stats}
    return output_stats


#%%
## MAIN FUNCTION
if __name__ == "__main__":
    
    ## DEFINING VERBOSITY
    verbose = True
    
    ####################################
    ### DEFINING DIRECTORY AND PATHS ###
    ####################################
    
    ## DEFAULT DIRECTORIES
    combined_database_path = r"R:\scratch\3d_cnn_project\combined_data_set"
    class_file_path = r"R:\scratch\3d_cnn_project\database\Experimental_Data\solvent_effects_regression_data.csv"
    
    ## DEFINING IMAGE FILE PATH
    image_file_path = r"R:\scratch\3d_cnn_project\images"
    ## DEFINING SIM PATH
    sim_path = r"R:\scratch\3d_cnn_project\simulations"
    
    ## DEFINING DATABASE PATH
    database_path=r"R:\scratch\3d_cnn_project\database"

    ## DEFINING LOCATION TO STORE PICKLE
    path_pickle = r"R:\scratch\3d_cnn_project\storage"
    
    ## CHECKING MULTIPLE PATHS
    combined_database_path, class_file_path, image_file_path, sim_path, database_path, path_pickle = \
            check_multiple_paths(combined_database_path, class_file_path, image_file_path, sim_path, database_path, path_pickle )

    
    #%%
    ## DEFINING MAIN DIR LIST
    main_dir_list=[
            # '190627-20_20_20_32ns_3_descriptor_extra_layer_cross_validation_solute-voxnet',
#            '190705-20_20_20_32ns_firstwith10_solute-solvent_net',
#            '190705-20_20_20_32ns_firstwith10_solute-voxnet',
#            '190705-20_20_20_20ns_firstwith10_cosolvent-solvent_net',
#            '190705-20_20_20_20ns_firstwith10_solute-solvent_net',
            # '190705-20_20_20_32ns_firstwith10_cosolvent-solvent_net',
            # '190702-32_32_32_32ns_planar_cosolvent-vgg16',
#             '190702-32_32_32_32ns_planar_solute-vgg16',
            # '190621-30_30_30_cross_validation_solute-solvent_net_first32ns',
            # '190621-30_30_30_cross_validation_cosolvent-solvent_net_first32ns',
            # '190621-30_30_30_cross_validation_cosolvent-voxnet_first32ns',
            # '190621-30_30_30_cross_validation_solute-solvent_net_first32ns',
            # '190621-30_30_30_cross_validation_solute-voxnet_first32ns',
            # '190626-20_20_20_32ns_3_descriptor_cross_validation_solute-solvent_net',
            # '190626-20_20_20_32ns_3_descriptor_cross_validation_cosolvent-solvent_net',
#             '190625-20_20_20_32ns_descriptor_cross_validation_solute-voxnet',
#             '190625-20_20_20_32ns_descriptor_cross_validation_solute-solvent_net',
#             '190625-20_20_20_32ns_descriptor_cross_validation_cosolvent-voxnet'
#             '190623-20_20_20_32ns_withoxy_cross_validation_solute-solvent_net',
#            '190621-30_30_30_cross_validation_cosolvent-solvent_net_first32ns',
#            '190621-30_30_30_cross_validation_cosolvent-voxnet_first32ns',
#            '190621-30_30_30_cross_validation_solute-solvent_net_first32ns',
#            '190621-30_30_30_cross_validation_solute-voxnet_first32ns',
            # ----
            # '190718-32_32_32_20ns_firstwith10-vgg16-solute_planar_cases',
            # '190718-32_32_32_20ns_firstwith10-vgg16-cosolvent_planar_cases',
            # '190709-32_32_32_20ns_firstwith10_cosolvent-solvent_net_Different_sizes',
            # '190709-16_16_16_20ns_firstwith10_cosolvent-solvent_net_Different_sizes',
            # '190718-cross_valid_oxy_rep-20_20_20_20ns_firstwith10_oxy-solvent_net-cosolvent',
            # '190709-32_32_32_20ns_firstwith10_solute-solvent_net_Different_sizes',
            # '190718-cross_valid_oxy_rep-20_20_20_20ns_firstwith10_oxy-solvent_net-cos'
            # r'190718-cross_valid_oxy_rep-20_20_20_20ns_firstwith10_oxy-solvent_net-solute',
#            r"190726-cross_valid-rerun-20_20_20_20ns_firstwith10_oxy-solvent_net-solute",
#            r"190726-cross_valid-rerun-20_20_20_20ns_firstwith10_oxy-solvent_net-cosolvent",
#            r"190725-cross_valid-newrep-20_20_20_20ns_oxy_3chan-solvent_net-solute",
#            r"190725-cross_valid-newrep-20_20_20_20ns_oxy_3chan-solvent_net-cosolvent"
            # r"190726-cross_valid-rerun-20_20_20_20ns_firstwith10_oxy-solvent_net-solute",
            # r"190727-oxy_cross_val_sizes-32_32_32_20ns_oxy_firstwith10-solvent_net-solute"
            # r"190727-oxy_cross_val_sizes-16_16_16_20ns_oxy_firstwith10-solvent_net-solute",
            
            #----
#            r"190725-cross_valid-newrep-20_20_20_20ns_oxy_3chan-solvent_net-solute",
#            r"190731-diffrep_cross_val-20_20_20_20ns_3channel_hydroxyl_firstwith10-solvent_net-cosolvent",
#            r"190731-diffrep_cross_val-20_20_20_20ns_3channel_hydroxyl_firstwith10-solvent_net-solute",
#            r"190731-diffrep_cross_val-20_20_20_20ns_solvent_only_firstwith10-solvent_net-cosolvent",
#            r"190731-diffrep_cross_val-20_20_20_20ns_solvent_only_firstwith10-solvent_net-solute",
            # r"190726-cross_valid-rerun-20_20_20_20ns_firstwith10_oxy-solvent_net-cosolvent",
            # r"190725-cross_valid-newrep-20_20_20_20ns_oxy_3chan-solvent_net-cosolvent",
            
#            r"190801-_cross_val-20_20_20_20ns_4chan_hydroxyl_firstwith10-solvent_net-cosolvent",
            # r"190801-_cross_val-20_20_20_20ns_4chan_hydroxyl_firstwith10-solvent_net-solute",
#            r"190730-3chan_cross_val-32_32_32_20ns_oxy_3chan_firstwith10-solvent_net-solute",
            
#            r"190730-3chan_cross_val-32_32_32_20ns_oxy_3chan_firstwith10-solvent_net-solute",
#            r"190805-cross_val_size-16_16_16_20ns_oxy_3chan_firstwith10-solvent_net-cosolvent",
#             r"190805-cross_val_size-16_16_16_20ns_oxy_3chan_firstwith10-solvent_net-solute",
            r"20200201-cross_val_size-20_20_20_20ns_oxy_3chan-voxnet-cosolvent",
            r"20200201-cross_val_size-20_20_20_20ns_oxy_3chan-voxnet-solute",
#        
            # r"190802-cross_val_planar-32_32_32_20ns_oxy_3chan_firstwith10-vgg16-solute",
            # r"190802-cross_val_planar-32_32_32_20ns_oxy_3chan_firstwith10-vgg16-cosolvent",
            # r"190730-3chan_cross_val-32_32_32_20ns_oxy_3chan_firstwith10-solvent_net-cosolvent",
            
        

            # '190709-16_16_16_20ns_firstwith10_solute-solvent_net_Different_sizes',
            # '190718-cross_valid_oxy_rep-20_20_20_20ns_firstwith10_oxy-solvent_net-solute',
            # '190718-cross_valid_oxy_rep-20_20_20_20ns_firstwith10_oxy-solvent_net-solute',
            # '190718-cross_valid-withMD-20_20_20_32ns_firstwith10-solvent_net-cosolvent', # <-- problem
            ] # 190718-cross_valid-withMD-20_20_20_32ns_firstwith
    
    ## DEFINING IF YOU WANT TO STORE PICKLE
    store_pickle = True
    
    for main_dir in main_dir_list:
        # main_dir = r"190617-cross_validation_solute-solvent_net_first32ns"
        # main_dir = r"190606-cross_validation_cosolvent_solvent_net_last40ns"

        ## DEFINING RESULTS PICKLE FILE
        results_pickle_file = r"model.results" # DIO 
        
        ## DEFINING CROSS VALIDATION FILE
        cross_validation_file="cross_valid.txt"
        
        ## DEFINING PATH TO SIM
        path_sim_dir = os.path.join(  sim_path, main_dir )
        
        ## DEFINING FULL PATH TO CROSS VALIDATION FILE
        path_cross_validation_file = os.path.join(  path_sim_dir,  cross_validation_file )
        
        ## LOGICAL TO WANT COMBINED PLOT
        want_combined_plot = False # True if you want combined plot
        ## SAVING FIGURE
        save_fig = True
        ## FIGURE EXTENSION
        fig_extension = "png" # "svg" # "eps"  # "png" # "eps"
        ## DEFINING FIGURE NAME
        fig_name = os.path.join(image_file_path, main_dir + '.' + fig_extension)
        ## FIGURE SIZE
        save_fig_size = (8, 8)
    
        # save_fig_size = (16.8/2, 16.8/2), # (16.8/3, 16.8/3),
        
        ## DEFINING INPUTS
        cross_valid_inputs = {
                'main_dir': main_dir,
                'combined_database_path': combined_database_path,
                'class_file_path': class_file_path,
                'image_file_path': image_file_path,
                'sim_path': sim_path,
                'database_path': database_path,
                'results_pickle_file': results_pickle_file,
                'verbose': verbose,
                }
    
        ## DEFINING PICKLE STORAGE NAME
        pickle_storage_name =  os.path.join( path_pickle, main_dir + "_storage.pickle")
        ## RUNNING CROSS VALIDATION RESULTS
        if os.path.isfile(pickle_storage_name) is not True or store_pickle is False:
            ## FINDING CROSS VALIDATION RESULTS
            cross_valid_results = analyze_cross_validation( **cross_valid_inputs )
            # cross_valid_results = None
            ## STORING PICKLE
            if store_pickle is True and cross_valid_results is not None:
                with open(pickle_storage_name, 'wb') as f:  # Python 3: open(..., 'wb')
                    pickle.dump([cross_valid_results], f, protocol=2)  # <-- protocol 2 required for python2   # -1
        ## RESTORING
        else:
            print("RELOADING RESULTS FROM %s"%(pickle_storage_name) )
            cross_valid_results = load_pickle_general(pickle_storage_name)[0]
        ## PLOTTING
        if sys.prefix != '/home/akchew/envs/cs760':
            ## DEFINING PLOTTING INPUTS
            parity_plot_inputs = \
                {
                        'sigma_act_label': 'y_true',
                        'sigma_pred_label': 'y_pred',
                        'sigma_pred_err_label': 'y_pred_std',
                        'mass_frac_water_label': 'mass_frac',
                        'save_fig_size': save_fig_size,
                        'save_fig': save_fig,
                        'fig_extension': fig_extension,
                        'fig_name': fig_name,
                        }
                
            ## PLOTTING ALL CROSS VALIDATIONS
            fig, ax = plot_all_cross_validations( cross_validation_storage = cross_valid_results.cross_validation_storage, 
                                        parity_plot_inputs = parity_plot_inputs,
                                        want_combined_plot = True)
    #%%
    
    #%%
    
#    ## GETTING TEST DATABASE
#    test_set_df_full = get_test_set_df_from_cross_valid(cross_valid_results)
#    
#    ## GETTING CUMULATIVE RMSE
#    cumulative_RMSE = compute_cumulative_rmse( test_set_df_full )
#    
#    
#
#        
#    
#    ## GETTING STATISTICS
#    output_stats = compute_stats_from_cross_valid(test_set_df_full = test_set_df_full, 
#                                                  desired_stats = ['slope', 'rmse', 'pearson_r']
#                                                  )
    
#    ## GETTING CUMULATIVE RMSE
#    cumulative_RMSE = compute_cumulative_rmse( cross_valid_results )
#    
#    #%%
#    n_bins = 30
#    ## PLOTTING
#    fig, ax = plt.subplots(figsize=(8, 4))
#    # plot the cumulative histogram
#    n, bins, patches = ax.hist(cumulative_RMSE, n_bins, density=True, histtype='step',
#                               cumulative=True)
