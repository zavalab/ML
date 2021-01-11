# -*- coding: utf-8 -*-
"""
extract_deep_cnn.py
This code runs extraction of deep cnns. 

Created on: 04/25/2019

Author(s):
    - Alex K. Chew (alexkchew@gmail.com)
    

"""
## IMPORTING NECESSARY MODULES
import os
## IMPORTING PANDAS
import pandas as pd
## IMPORTING NUMPY
import numpy as np
## IMPORTING PICKLE
import pickle
import sys

## IMPORT SCIPY
import scipy as sp

## IMPORTING KERAS DETAILS
from keras.layers import Conv3D, MaxPool3D, Flatten, Dense
from keras.layers import Dropout, Input, BatchNormalization
from keras.losses import mean_squared_error
from keras.optimizers import Adam
from keras.models import Model
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint

## IMPORTING GLOBAL VARIABLES
from core.global_vars import SOLUTE_TO_TEMP_DICT, CNN_DICT, SAMPLING_DICT
## CHECKING TOOLS
from core.check_tools import check_testing
## IMPORTING COMBINING ARRAYS
from combining_arrays import combine_instances
## IMPORTING PATH FUNCTIONS
from core.path import find_paths

## IMPORTING NOMENCLATURE
from core.nomenclature import read_combined_name, extract_representation_inputs, extract_sampling_inputs

#######################
### 3D CNN NETWORKS ###
#######################
## IMPORTING SOLVENT NET
from deep_cnn_solvent_net_3 import cnn as solvent_net
## IMPORTING ORION
from deep_cnn_ORION import cnn as orion
## IMPORTING VOXNET
from deep_cnn_vox_net import vox_cnn as voxnet
## IMPORTING VGG16
from deep_cnn_vgg16 import cnn as vgg16
## TAKING EXTRACTION SCRIPTS
from extraction_scripts import load_pickle_general
## IMPORTING ML FUNCTIONS
from core.ml_funcs import get_list_args
## IMPORTING TRAIN DEEP CNN
from train_deep_cnn import train_deep_cnn
## IMPORTING ANALYSIS
from analyze_deep_cnn import analyze_deep_cnn


#%%
## MAIN FUNCTION
if __name__ == "__main__":

    ## SEE IF TESTING IS ON
    testing = check_testing()
    
    ## DEFINING DICTIONARY FOR CNN DICT
    cnn_dict = CNN_DICT
    
    ## SEEING IF TESTING IS TRUE
    if testing == True:
    
        ## DEFINING SOLVENT LIST
#        solvent_list = [ 'DIO' ]# 'GVL', 'THF' ] , 'GVL', 'THF'
#        ## DEFINING MASS FRACTION DATA
#        mass_frac_data = ['10'] # , '25', '50', '75'
#        
        solvent_list = [ 'DIO', 'GVL',  'THF']# 'GVL', 'THF' ] , 'GVL', 'THF'
        ## DEFINING MASS FRACTION DATA
        mass_frac_data = ['10', '25', '50', '75'] # , '25', '50', '75'
        
        ## DEFINING SOLUTE LIST
        solute_list = list(SOLUTE_TO_TEMP_DICT)
        ## DEFINING TYPE OF REPRESENTATION
        representation_type = 'split_avg_nonorm' # split_avg_nonorm split_average # split_avg_nonorm_planar
        representation_inputs = {
                'num_splits': 10
                }
        
        ## DEFINING DATA TYPE
        # data_type="30_30"
        data_type="20_20_20_20ns_oxy_3chan"
        
        ## DEFINING SIMULATION DIRECTORY
        sim_dir = r"2020203-5fold_train_20_20_20_20ns_oxy_3chan"
        # r"20200131-5fold_train_20_20_20_20ns_oxy_3chan"
        
        ## DEFINING SPECIFIC DIRECTORY
        specific_dir = r"20_20_20_20ns_oxy_3chan-split_avg_nonorm-10-strlearn-0.80-voxnet-500-CEL_ETBE_FRU_LGA_PDO_XYL_tBuOH-10_25_50_75-DIO_GVL_THF"
        # r"20_20_20_20ns_oxy_3chan-split_avg_nonorm-10-strlearn-0.80-solvent_net-500-CEL_ETBE_FRU_LGA_PDO_XYL_tBuOH-10_25_50_75-DIO_GVL_THF"
        
        ## DEFINING PATHS
        # database_path = r"R:\scratch\SideProjectHuber\Analysis\CS760_Database_32_32_32" # None # Since None, we will find them!
        database_path = os.path.join( r"R:\scratch\3d_cnn_project\database", data_type)  # None # Since None, we will find them!        
        class_file_path = r"R:\scratch\3d_cnn_project\database\Experimental_Data\solvent_effects_regression_data.csv"
        combined_database_path = r"R:\scratch\3d_cnn_project\combined_data_set"
        output_file_path = os.path.join( r"R:\scratch\3d_cnn_project\simulations",
                                         sim_dir,
                                         specific_dir)# OUTPUT PATH FOR CNN NETWORKS
        results_file_path = None
        
        ## DEFINING VERBOSITY
        verbose = True
        
        ## DEFINING RETRAINING DETAILS
        retrain = False
        
        ## SELECTING TYPE OF CNN
        cnn_type = 'voxnet' # 'orion' # 'solvent_net' 'voxnet'
        '''
        voxnet
        orion
        solvent_net
        '''
        
        ## DEFINING NUMBER OF EPOCHS
        # cnn_dict['epochs'] = 1
        
        ## LOGICAL FOR STORING AND RELOADING TEST PICKLES
        want_training_test_pickle = False
        want_basic_name = True
        want_augment= True
        want_descriptors = False
        
        ## DEFINING NUMBER OF CROSS VALIDATION FOLDING
        num_cross_validation_folds = 5
        
        ## DEFINING SAMPLING DICTIONARY
        sampling_dict = { 
            'name': 'strlearn',
            'split_percentage': 0.80, # 3,
            }
    else:
        ## ADDING OPTIONS 
        from optparse import OptionParser # for parsing command-line options
        ## RUNNING COMMAND LINE PROMPTS
        use = "Usage: %prog [options]"
        parser = OptionParser(usage = use)
        
        ## REPRESENTATION TYPE
        parser.add_option('-r', '--representation', dest = 'representation_type', help = 'Representation type', default = 'split_average', type=str)
        parser.add_option("-g", "--representation_inputs", dest="representation_inputs", action="callback", type="string", callback=get_list_args,
                  help="For multiple inputs, simply separate by comma (no whitespace)", default = 5)
        ## MASS FRACTIONS
        parser.add_option("-m", "--massfrac", dest="mass_frac_data", action="callback", type="string", callback=get_list_args,
                  help="For multiple mass fractions, separate each solute name by comma (no whitespace)", default = ['10', '25', '50', '75'])
        ## SOLVENT NAMES
        parser.add_option("-x", "--solvent", dest="solvent_list", action="callback", type="string", callback=get_list_args,
                  help="For multiple solvents, separate each solute name by comma (no whitespace)", default = [ 'DIO', 'GVL', 'THF' ] )
        ## SOLUTE NAMES
        parser.add_option("-s", "--solute", dest="solute_list", action="callback", type="string", callback=get_list_args,
                  help="For multiple solutes, separate each solute name by comma (no whitespace)", default = None)
        
        ## MODEL TYPE
        parser.add_option("-q", "--cnntype", dest="cnn_type", type=str, help="Type of cnn (e.g. voxnet, solventnet, orion)", default = 'voxnet')
        
        ## NUMBER OF EPOCHS
        parser.add_option("-n", "--epochs", dest="epochs", type=int, help="Number of epochs", default = 500)
        
        ## DIRECTORY LOCATIONS
        parser.add_option('-d', '--database', dest = 'database_path', help = 'Full path to database', default = None)
        parser.add_option('-c', '--classfile', dest = 'class_file_path', help = 'Full path to class csv file', default = None)
        parser.add_option('-a', '--combinedfile', dest = 'combined_database_path', help = 'Full path to combined pickle directory', default = None)
        parser.add_option('-o', '--outputfile', dest = 'output_file_path', help = 'Full path to output weights and pickles', default = None)
        parser.add_option('-w', '--resultsfile', dest = 'results_file_path', help = 'Full path to pickle after analysis', default = None)
        parser.add_option("-v", dest="verbose", action="store_true", )
        
        ## DEFINING DATA SET TYPE
        parser.add_option('-z', '--datatype', dest = 'data_type', help = 'data type', type="string", default = "20_20_20")
        
        ## DEFINING RETRAINING IF NECESSARY
        parser.add_option("-t", dest="retrain", action="store_true", default = False )
        
        ## STORING TRAINING PICKLE
        parser.add_option("-p", dest="want_training_test_pickle", action="store_true", default = False )
        parser.add_option("-b", dest="want_basic_name", action="store_true", default = False )
        
        ## WANT DESCRIPTORS
        parser.add_option("--want_descriptors", dest="want_descriptors", action="store_true", default = False )
        
        ## REPRESENTATION TYPE
        parser.add_option('--samplingtype', dest = 'sampling_type', help = 'Sampling type', default = 'split_average', type=str)
        parser.add_option("--samplinginputs", dest="sampling_inputs", action="callback", type="string", callback=get_list_args,
                  help="For multiple inputs, simply separate by comma (no whitespace)", default = 5)
        
        ## AUGMENTATION
        parser.add_option("--no_augment", dest="want_augment", action="store_false", default = True ) 
        
        ## CROSS VALIDATION FOLDS
        parser.add_option("--num_cross_folds", dest="num_cross_validation_folds", help = "Number of cross validation folds", type=int, default = 1 ) 
        
        ### GETTING ARGUMENTS
        (options, args) = parser.parse_args() # Takes arguments from parser and passes them into "options" and "argument"
        
        ### DEFINING ARUGMENT
        solute_list = options.solute_list
        solvent_list = options.solvent_list
        mass_frac_data = options.mass_frac_data
        ## REPRESENTATION
        representation_type = options.representation_type
        representation_inputs = options.representation_inputs
        
        ## DEFINING MODEL TYPE
        cnn_type = options.cnn_type
        
        ## AUGMENTATION
        want_augment = options.want_augment
        
        ## DEFINING DATA TYPE
        data_type = options.data_type
        
        ## FILE PATHS
        database_path = options.database_path
        class_file_path = options.class_file_path
        combined_database_path = options.combined_database_path
        output_file_path = options.output_file_path
        results_file_path = options.results_file_path
        
        ## DEFINING SAMPLING INPUTS
        sampling_type = options.sampling_type
        sampling_inputs = options.sampling_inputs
        
        ## DEFINING CROSS VALIDATION FOLDS
        num_cross_validation_folds = options.num_cross_validation_folds
        
        ## SEEING IF FILE PATHS
        if database_path == "None":
            database_path = None
        if class_file_path == "None":
            class_file_path = None
        if combined_database_path == "None":
            combined_database_path = None
        if output_file_path == "None":
            output_file_path = None
        if results_file_path == "None":
            results_file_path = None
        
        ## VERBOSITY
        verbose = options.verbose
        
        ## RETRAIN
        retrain = options.retrain
        
        ## TRAINING TEST PICKLE
        want_training_test_pickle = options.want_training_test_pickle
        want_basic_name = options.want_basic_name
        want_descriptors = options.want_descriptors
        
        ## ADDING EPOCHS
        cnn_dict['epochs'] = options.epochs
        
        ## UPDATING REPRESATION INPUTS
        representation_inputs = extract_representation_inputs( representation_type = representation_type, 
                                                              representation_inputs = representation_inputs )
        
        ## UPDATING SAMPLING INPUTS
        sampling_dict = extract_sampling_inputs( sampling_type = sampling_type, 
                                                 sampling_inputs = sampling_inputs,)

    ## LOADING THE DATA
    instances = combine_instances(
                     solute_list = solute_list,
                     representation_type = representation_type,
                     representation_inputs = representation_inputs,
                     solvent_list = solvent_list, 
                     mass_frac_data = mass_frac_data, 
                     verbose = verbose,
                     database_path = database_path,
                     class_file_path = class_file_path,
                     combined_database_path = combined_database_path,
                     data_type = data_type,
                     )

    ### TRAINING CNN
    deep_cnn = train_deep_cnn(
                     instances = instances,
                     sampling_dict = sampling_dict,
                     cnn_type = cnn_type,
                     cnn_dict = cnn_dict,
                     retrain=retrain,
                     output_path = output_file_path,
                     class_file_path = class_file_path,
                     verbose = verbose,
                     want_training_test_pickle = want_training_test_pickle,
                     want_basic_name = want_basic_name,
                     want_descriptors = want_descriptors,
                     want_augment = want_augment,
                     num_cross_validation_folds = num_cross_validation_folds,
                     )
    
    #%%
    
    
    
    ### ANALYZING DEEP CNN
    analysis = analyze_deep_cnn( instances = instances, 
                                 deep_cnn = deep_cnn )
    
    ## STORING ANALYSIS
    analysis.store_pickle( results_file_path = results_file_path )
    
    
    
