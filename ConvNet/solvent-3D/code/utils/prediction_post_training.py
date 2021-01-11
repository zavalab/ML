# -*- coding: utf-8 -*-
"""
prediction_post_training.py
The purpose of this script it to make predictions of a data set given that you 
have completely trained a neural network. 

Created on: 06/12/2019

Author(s):
    - Alex K. Chew (alexkchew@gmail.com)
"""

## IMPORTING OS
import os
import pandas as pd
import glob
import matplotlib.pyplot as plt
import numpy as np
import sys

## IMPORTING FUNCTIONS
from core.import_tools import read_file_as_line
## IMPORTING GLOBAL VARIABLES
from core.global_vars import SOLUTE_TO_TEMP_DICT, CNN_DICT, SAMPLING_DICT
## IMPORTING NOMENCLATURE
from core.nomenclature import read_combined_name

## CHECKING TOOLS
from core.check_tools import check_testing

#######################
### 3D CNN NETWORKS ###
#######################
## IMPORTING COMBINING ARRAYS
from combining_arrays import combine_instances
## IMPORTING TRAIN DEEP CNN
from train_deep_cnn import train_deep_cnn
## IMPORTING ANALYSIS TOOL
from analyze_deep_cnn import analyze_deep_cnn, metrics
## FUNCTION TO LOAD MODEL
from keras.models import load_model

## LOADING LOCATING INSTANCES
from core.ml_funcs import locate_test_instance_value

## DEFINING FUNCTION TON PLOT
from read_extract_deep_cnn import plot_parity_publication_single_solvent_system

## IMPORTING PATHS
from core.path import read_combined_name_directories, extract_combined_names_to_vars, extract_input_MD_name

    
## TAKING EXTRACTION SCRIPTS
from extraction_scripts import load_pickle,load_pickle_general

## LOADING COMBINE TRAINING DATA
from combining_arrays import combine_training_data

## DEFINING PATH TO DATABASE
PATH_DATABASE = r"R:\scratch\3d_cnn_project\database"
# PATH_DATABASE = r"/Volumes/akchew/scratch/3d_cnn_project/database"
## DEFINING PATH TO EXPERIMENTAL DATA
PATH_EXP_DATA = os.path.join(PATH_DATABASE, "Experimental_Data")

## DEFINING NAME
# TEST_DATABASE_BASENAME=r"20_20_20_32ns_first"
# TEST_DATABASE_BASENAME=r"20_20_20_32ns_firstwith10_"
TEST_DATABASE_BASENAME=r"20_20_20_20ns_oxy_3chan_"

### FUNCTION TO GET TEST DATABASE DICT
def get_test_pred_test_database_dict(test_database_basename = TEST_DATABASE_BASENAME):
    '''
    The purpose of this function is to get database dict based on basename
    INPUTS:
        test_database_basename: [str]
            database name, e.g.:
                20_20_20_20ns_firstwith10_
    '''
    ## LINKING DATABASES
    test_database_dict = \
            {
                'DMSO': {
                        'path_database': os.path.join( PATH_DATABASE, test_database_basename + "dmso"),
                        'path_exp_data': os.path.join( PATH_EXP_DATA, "solvent_effects_regression_data_DMSO.csv"),
                        },
                'ACN': {
                        'path_database': os.path.join( PATH_DATABASE, test_database_basename + "ACN"),
                        'path_exp_data': os.path.join( PATH_EXP_DATA, "solvent_effects_regression_data_ACN.csv"),
                        },
                'ACE': {
                        'path_database': os.path.join( PATH_DATABASE, test_database_basename + "ACE"),
                        'path_exp_data': os.path.join( PATH_EXP_DATA, "solvent_effects_regression_data_ACE.csv"),
                        },
            }
    
    ## CHECKING IF DATABASE EXISTS
    for each_key in test_database_dict:
        if os.path.isdir(test_database_dict[each_key]['path_database']) is False:
            print("Warning! We do not find the database path")
            print("Please check database path:")
            print(test_database_dict[each_key]['path_database'])
            sys.exit(1)
    
    return test_database_dict

## LINKING DATABASES
TEST_DATABASE_DICT = \
        {
            'NAT_CATAL_DMSO': {
                    'path_database': os.path.join( PATH_DATABASE, TEST_DATABASE_BASENAME + "dmso"),
                    'path_exp_data': os.path.join( PATH_EXP_DATA, "solvent_effects_regression_data_DMSO.csv"),
                    },
            'NAT_CATAL_ACN': {
                    'path_database': os.path.join( PATH_DATABASE, TEST_DATABASE_BASENAME + "ACN"),
                    'path_exp_data': os.path.join( PATH_EXP_DATA, "solvent_effects_regression_data_ACN.csv"),
                    },
            'ALI_ACE': {
                    'path_database': os.path.join( PATH_DATABASE, TEST_DATABASE_BASENAME + "ACE"),
                    'path_exp_data': os.path.join( PATH_EXP_DATA, "solvent_effects_regression_data_ACE.csv"),
                    },
        }

############################################
### CLASS FUNCTION TO RELOAD AND PREDICT ###
############################################
class predict_with_trained_model:
    '''
    The purpose of this function is to make predictions of sigma given a trained model. 
    INPUTS:
        path_model: [list]
            list to model paths
    OUTPUTS:
        
    '''
    def __init__(self,
                 path_model,
                 verbose = False,
                 ):
        
        ## STORING VARIABLES
        self.path_model = path_model
        self.verbose = verbose
        
        ## FINDING MODEL DETAILS
        self.find_model_details()
        
        ## LOADING PREDICTIVE MODEL
        self.load_pred_model()
        
        return
    ### FUNCTION TO FIND DETAILS ABOUT THE MODEL
    def find_model_details(self,):
        '''
        The purpose of this function is to find details about the model by finding 
        the directory the model is in and extracting information about it. Note 
        that the model it uses is the first model
        INPUTS:
            self
        OUTPUTS:
            self.specific_sim_dir: [str]
                specific name of the simulation directory
            self.current_directory_extracted: [dict]
                dictionary extracting the simulation directory
            ## DETAILS ABOUT THE MODEL
            self.representation_type: [str]
                representation types
            self.representation_inputs: [list]
                representation inputs
            self.sampling_dict: [dict]
                sampling details
            self.data_type: [str]
                type of data used
            self.cnn_type: [str]
                type of cnn used
            self.num_epochs: [int]
                number of epochs
            self.solute_list: [list]
                list of solutes used
            self.solvent_list: [list]
                list of solvents used
            self.mass_frac_data: [list]
                list of mass fractions used            
        '''
        ## FINDING SPECIFIC DIRECTORY
        self.specific_sim_dir = os.path.basename(os.path.dirname(self.path_model[0]))
        
        ## EXTRACTING DIRECTORY INFORMATION
        self.current_directory_extracted = read_combined_name( self.specific_sim_dir )
        
        ## EXTRACTING INFORMATION
        self.representation_type, \
        self.representation_inputs, \
        self.sampling_dict, \
        self.data_type, \
        self.cnn_type, \
        self.num_epochs, \
        self.solute_list, \
        self.solvent_list, \
        self.mass_frac_data, \
        self.want_MD_descriptors = extract_combined_names_to_vars(extracted_name = self.current_directory_extracted)
        
        return
    
    ### FUNCTION TO LOAD MODEL
    def load_pred_model(self):
        ''' Function that loads the model '''
        ## LOADING ALL MODELS
        self.model = []
        ## LOOPING
        for each_model in self.path_model:
            ## DEFINING CURRENT MODEL
            current_model = load_model(each_model)
            self.model.append(current_model)
            ## PRINTING SUMMARY
            if self.verbose is True:
                print("Loading model: %s"%(each_model))
                # self.model.summary()
        return
    
    ### FUNCTION TO MAKE PREDICTIONS
    def predict_test_set(self, 
                         list_of_directories = [],
                         path_test_database = None,
                         num_partitions = 2,
                         want_override_combine_training = True,
                         ):
        '''
        The purpose of this function is to predict the test set using the trained 
        model. 
        INPUTS:
            path_test_database: [str]
                full path to the test set database
            num_partitions: [int]
                number of partitions to use, default is zero
            want_override_combine_training: [logical, default=True]
                True if you want to override using the training data representation as a 
                testing representation. It will override with the number of partitions, which should be 2.
        OUTPUTS:
            stored_predicted_value_list: [list]
                list of predicted values
        '''
        ## FINDING ALL FILES
        if path_test_database is not None:
            directory_paths, directory_basename, directory_extracted_names = read_combined_name_directories( 
                                                                                    path = path_test_database,
                                                                                    extraction_func = extract_input_MD_name,
                                                                                    want_dir = False,
                                                                                    want_single_path = False,
                                                                                    )
        else:
           directory_paths = list_of_directories[:]
        
        ## DEFINING LIST
        stored_predicted_value_list = []
        
        ## DEFINING REPRSENTATION INPUTS
        representation_inputs = self.representation_inputs.copy()
        # {**self.representation_inputs}
        ## SEEING IF OVERRIDE IS DESIRED
        if want_override_combine_training is True:
            print("Overriding representation inputs for splitting")
            print("Splitting partitions: %d"%(num_partitions) )
            representation_inputs['num_splits'] = num_partitions
        
        else:
            print("Using training representation as testing")
        
        ## PRINTING REPRESENTATION INPUTS
        print("Representation inputs:")
        print(representation_inputs)
        
        ## STORING THE DIRECTORY
        # predicted_storage = {}
        
        ## LOOPING THROUGH EACH PATH
        for idx, full_train_pickle_path in enumerate(directory_paths):
            
            ## SEEING IF PATH IS DESIRED
            if path_test_database is None:
                ## EXTRACTING NAMES
                directory_paths, directory_basename, directory_extracted_names = read_combined_name_directories( 
                                                                                        path = full_train_pickle_path,
                                                                                        extraction_func = extract_input_MD_name,
                                                                                        want_dir = False,
                                                                                        want_single_path = True,
                                                                                        )
                ## SETTING IDX TO ZERO
                idx = 0
                
            ## DEFINING EXTRACTED NAME
            current_extracted_name = directory_extracted_names[idx]
            
            ## DEFINING CURRENT BASE NAME
            current_basename = directory_basename[idx]
            
            ## PRINTING
            print("Working on: %s"%(current_basename) )
            
            ## LOADING THE PICKLE FOR INSTANCE INFORMATION
            instance_data = load_pickle(full_train_pickle_path)

            ## CHANGING TRAINING DATA INSTANCE REPRESENTATION
            instance_respresentation, str_output = combine_training_data( training_data_for_instance = instance_data,
                                                                          representation_type = self.representation_type,
                                                                          representation_inputs = representation_inputs)
    
            ## CONVERTING TO ARRAY (MAY BE SLOW)
            instance_respresentation = np.asarray(instance_respresentation)[:num_partitions] # First two instances
            
            ## MAKING PREDICTIONS FOR EACH MODEL
            predicted_y = [each_model.predict(instance_respresentation) for each_model in self.model]
            
            ## STORING
            # predicted_storage[full_train_pickle_path] = predicted_y[:]
            
            ## DEFINING PREDICTED VALUES AVERAGED
            predicted_y_combined = np.mean(predicted_y, axis = 0)
            
            ## FINDING AVERAGE AND STD
            predicted_y_avg = np.mean(predicted_y_combined)
            predicted_y_std = np.std(predicted_y_combined)
            
            ## CREATING DICTIONARY
            output_y =  current_extracted_name.copy()
            
            ## STORING
            output_y['predicted_y'] = predicted_y
            output_y['y_pred'] = predicted_y_avg
            output_y['y_pred_std'] = predicted_y_std
            output_y['directory_basename'] = current_basename
            
            ## APPENDING
            stored_predicted_value_list.append( output_y )
            
        return stored_predicted_value_list
    
    ### FUNCTION TO PLOT PARITY PLOT
    @staticmethod
    def plot_test_set_parity(stored_predicted_value_list, parity_plot_inputs):
        '''
        The purpose of this function is to plot the test set parity
        '''
        ## CREATING DATAFRAME
        df = pd.DataFrame( stored_predicted_value_list )
        
        ## PLOTTING PARITY
        fig, ax = plot_parity_publication_single_solvent_system( dataframe = df,
                                                       **parity_plot_inputs)
        return fig, ax
        
    
    ### FUNCTION TO TAG ON TEST INSTANCE TRUE VALUES
    @staticmethod        
    def add_test_set_exp_values( stored_predicted_value_list, path_exp_data ):
        '''
        The purpose of this function is to add test set experimental values. 
        INPUTS:
            stored_predicted_value_list: [list]
                list of predicted values
            path_exp_data: [str]
                path to experimental data
        OUTPUTS:
            null -- should modify stored_predicted_value_list
        '''
        ## READING CSV FILE
        csv_file = pd.read_csv( path_exp_data )
        
        ## LOOPING THROUGH EACH AND LOCATING TEST INSTANCE
        for idx, each_predicted in enumerate(stored_predicted_value_list):
            ## DEFINING CURRENT BASE NAME
            current_basename = each_predicted['directory_basename']  # directory_basename[idx]
            ## PRINTING
            print("Adding experimental data for: %s" %(current_basename) )
            ## LOCATING TEST INSTANCE
            value = locate_test_instance_value( csv_file = csv_file,
                                                solute = each_predicted['solute'],
                                                cosolvent = each_predicted['cosolvent'],
                                                mass_frac_water = each_predicted['mass_frac'] ,
                                                temp = each_predicted['temp'],
                                               )
            ## STORE INTO INSTANCE
            each_predicted['y_true'] = value
        return
    

#%%
## MAIN FUNCTION
if __name__ == "__main__":
    
    ## SEE IF TESTING IS ON
    testing = check_testing()
    
    ####################################
    ### DEFINING DIRECTORY AND PATHS ###
    ####################################
    
    ## SEEING IF TESTING IS TRUE
    if testing == True:
    
        ## DEFINING SIM PATH
        sim_path = r"R:\scratch\3d_cnn_project\simulations"
        ## DEFINING MAIN DIRECTORY
        main_dir = "190725-newrep_20_20_20_20ns_oxy_3chan"
        
        ## DEFINING SPECIFIC SIMULATION DIR
        specific_sim_dir = "20_20_20_20ns_oxy_3chan-split_avg_nonorm-10-strlearn-0.80-solvent_net-500-CEL_ETBE_FRU_LGA_PDO_XYL_tBuOH-10_25_50_75-DIO_GVL_THF"
        
        ## DEFINING PATH TO SIM
        path_sim_dir = os.path.join(  sim_path, main_dir, specific_sim_dir )
        
        ## DEFINING MODEL WEIGHTS
        model_weights = "model.hdf5"
        ## DEFINING FULL PATH TO MODEL
        path_model = os.path.join(path_sim_dir, model_weights)
        
        ## DEFINING LOCATION OF FILES
        main_sim_loc = r"R:/scratch/SideProjectHuber/Simulations/191010-Cyclohexanol_4ns"
        ## DEFINING SPECIFIC LOCATION OF LIST
        ## DEFINING LIST
        sim_list = r"SolventNet_list.csv"
        
        ## DEFINING NUMBER OF PARTITIONS TO TEST
        num_partitions = 2
        
        ## DEFINING DATABASE
        test_database_basename = specific_sim_dir.split('-')[0] + '_'
        
        ## GETTING DATABASE DICT
        database_dict = get_test_pred_test_database_dict(test_database_basename = test_database_basename)
        
        ## DEFINING PATH TO TEST DATABASE
        path_test_database = None
        
        ## DEFINING OUTPUT CSV FILE
        output_csv="SolventNet_output.csv"
        
    else:
        
        ## ADDING OPTIONS 
        from optparse import OptionParser # for parsing command-line options
        ## RUNNING COMMAND LINE PROMPTS
        use = "Usage: %prog [options]"
        parser = OptionParser(usage = use)
        
        ## MODEL DETAILS
        parser.add_option('--path_model', dest = 'path_model', help = 'Path to model', default = '', type=str)
        ## NUMBER OF PARTITIONS
        parser.add_option('--num_partitions', dest = 'num_partitions', help = 'Number of partitions to divide up your dataset', default = 2, type=int)
        ## OUTPUT CSV
        parser.add_option('--output_csv', dest = 'output_csv', help = 'Output to csv file', default = "output.csv", type=str)
        ## MAIN SIM
        parser.add_option('--main_sim_loc', dest = 'main_sim_loc', help = 'Main simulation location', default = "", type=str)        
        ## INPUT CSV LIST
        parser.add_option('--sim_list', dest = 'sim_list', help = 'Sim list with pickle files', default = "", type=str)
        ## INPUT CSV LIST
        parser.add_option('--path_test_database', dest = 'path_test_database', help = 'Path to database', default = None, type=str)        
        ### GETTING ARGUMENTS
        (options, args) = parser.parse_args() # Takes arguments from parser and passes them into "options" and "argument"
        
        ## DEFINING VARIABLES
        path_model = options.path_model
        num_partitions = options.num_partitions
        output_csv = options.output_csv
        main_sim_loc = options.main_sim_loc
        sim_list = options.sim_list
        path_test_database = options.path_test_database
    
    ## DEFINING INPUTS FOR PREDICTED MODEL
    inputs_predicted_model = {
            'path_model': path_model,
            'verbose': True,
            }
    ## LOADING MODEL
    trained_model = predict_with_trained_model( **inputs_predicted_model
                                               )
    
    ### MAIN FUNCTIONS
    
    ## READING FILE
    sim_list_file = read_file_as_line( os.path.join(main_sim_loc,sim_list)  )
    
    ## DEFINING LIST OF DIRECTORIES
    list_of_directories = [ os.path.join(main_sim_loc, each_pickle) for each_pickle in sim_list_file ]
    
    ## PREDICTING THE VALUES
    stored_predicted_value_list = trained_model.predict_test_set(
                                                                    path_test_database = path_test_database,
                                                                    list_of_directories = list_of_directories,
                                                                    num_partitions = num_partitions,
                                                                    want_override_combine_training  = True,
                                                                    )
    
    ## DEFINING PATH
    path_to_output=os.path.join(main_sim_loc,output_csv)
    
    
    ## CREATING DATAFRAME
    df = pd.DataFrame( stored_predicted_value_list )
    
    ## OUTPUTTING TO CSV
    df.to_csv( path_to_output )
    
    print("Output to: %s"%(path_to_output))
    
    #%%
#    
#    
#    #%%
#    
#    ## DEFINING TYPE
#    # database_type = 'ACE' # 'DMSO'
#    # 'ACN' 'ACE'
#    
#    ## DEFINING DATABASE
#    # path_test_database = database_dict[database_type]['path_database']
#    
#    ## DEFINING PATH TEST DATABASE
#    path_test_database =  os.path.join( PATH_DATABASE, "20_20_20_2ns_oxy_3chan_FRU_HMF_DMSO") 
#    
#    ## DEFINING NUMBER OF PARTITIONS TO TEST
#    num_partitions = 1
##    ## DEFINING PARITY PLOT INPUTS    
##    parity_plot_inputs = \
##        {
##                'sigma_act_label': 'y_true',
##                'sigma_pred_label': 'y_pred',
##                'sigma_pred_err_label': 'y_pred_std',
##                'mass_frac_water_label': 'mass_frac',
##                }
#    
#    ## PREDICTING THE VALUES
#    stored_predicted_value_list = trained_model.predict_test_set(
#                                                                    path_test_database = path_test_database,
#                                                                    num_partitions = num_partitions,
#                                                                    want_override_combine_training  = True,
#                                                                    )
#    
#    #%%
#    
#    ## CREATING DATAFRAME
#    df = pd.DataFrame( stored_predicted_value_list )
#    
#    ## OUTPUTTING TO CSV
#    df.to_csv( r"C:\Users\akchew\Box Sync\VanLehnGroup\0.Manuscripts\Topics_in_catalysis_invited_paper\Excel_spreadsheet\csv_files\FRU_HMF_output.csv" )
#    
#    
#    
#    
#    #%%
#    ## DEFINING PATH TO EXPERIMENTAL DATA
#    path_exp_data = database_dict[database_type]['path_exp_data']
#    ## ADDING TEST SET EXPERIMENTAL VALUES
#    trained_model.add_test_set_exp_values( stored_predicted_value_list = stored_predicted_value_list,
#                             path_exp_data = path_exp_data)
#    
#    
#    #%%
#    ## PLOTTING FIGURES
#    fig, ax = trained_model.plot_test_set_parity(stored_predicted_value_list = stored_predicted_value_list, 
#                                                 parity_plot_inputs = parity_plot_inputs,)
#    
