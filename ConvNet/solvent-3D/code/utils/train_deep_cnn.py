# -*- coding: utf-8 -*-
"""
train_deep_cnn.py
The purpose of this script is to run the deep cnn network. Here, we will need 
machine learning modules, such as Keras (tensorflow). First, we will load the 
data, then run machine learning algorithms, then save the weights and any additional 
information.

Created on: 04/22/2019

Author(s):
    - Alex K. Chew (alexkchew@gmail.com)
    - Shengli Jiang (sjiang87@wisc.edu)
    - Weiqi
    
FUNCTIONS:
    augment_data: 
        augments the data set
    get_indices_for_k_fold_cross_validation: 
        gets indices for k-fold cross validations
    
"""
## IMPORTING NECESSARY MODULES
import os
## IMPORTING PANDAS
import pandas as pd
## IMPORTING NUMPY
import numpy as np
## IMPORTING PICKLE
import pickle
## IMPORTING SYSTEM
import sys

## IMPORT SCIPY
import scipy as sp

## IMPORTING TIME
import time

## IMPORTING KERAS DETAILS
from keras.layers import Conv3D, MaxPool3D, Flatten, Dense
from keras.layers import Dropout, Input, BatchNormalization
from keras.losses import mean_squared_error
from keras.optimizers import Adam
from keras.models import Model
from keras.utils import to_categorical
from keras.layers import concatenate
## COMPILE AGAIN WITH STORED WEIGHTS
from keras.models import load_model
from keras.callbacks import ModelCheckpoint

## IMPORTING GLOBAL VARIABLES
from core.global_vars import SOLUTE_TO_TEMP_DICT, INPUTS_FOR_DESCRIPTOR_FXN
## CHECKING TOOLS
from core.check_tools import check_testing
## IMPORTING COMBINING ARRAYS
from combining_arrays import combine_instances
## IMPORTING PATH FUNCTIONS
from core.path import find_paths
## IMPORTING NOMENCLATURE
from core.nomenclature import read_combined_name, extract_representation_inputs

## K FOLD MODEL SELECTION
from sklearn.model_selection import KFold

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
from core.ml_funcs import get_list_args, get_split_index_of_list_based_on_percentage
## IMPORTING NOMENCLATURE
from core.nomenclature import extract_instance_names
## IMPORTING METRICS TOOL
from core.ml_funcs import metrics
######################
### MD DESCRIPTORS ###
######################
from deep_cnn_md_descriptors import md_descriptor_network, get_descriptor_inputs

### FUNCTION TO CONVERT TIME
def convertSeconds(seconds):
    h = seconds//(60*60)
    m = (seconds-h*60*60)//60
    s = seconds-(h*60*60)-(m*60)
    return [h, m, s]

### FUNCTION TO GET INDICES FOR K-FOLD LEARNING
def get_indices_for_k_fold_cross_validation( instances, n_splits = 5, verbose = False ):
    '''
    The purpose of this function is to get the k-fold cross validation index 
    for cross validation training. 
    INPUTS:
        instances: [obj]
            instances object
        n_splits: [int, default = 5]
            number of splits for your x, y data
        verbose: [logical, default = False]
            print out details of splitting
    OUTPUTS:
        indices_dict: [dict]
            dictionary containing indices that you need to distinguish 
            training and validation/test set. 
        ## TO GET THE TRAINING AND VALIDATION INDICES
        x_train, x_val = x[train_index], x[test_index]
        y_train, y_val = y[train_index], y[test_index]
            
    '''
    ## DEFINING X AND Y
    x = np.array(instances.x_data)
    y = np.array(instances.y_label)
    
    ## DEFINING NAMES
    names = np.array(instances.instance_names)
    
    ## CROSS VALIDATION
    skf = KFold(n_splits=n_splits, random_state=0, shuffle=True)
    skf.get_n_splits(x, y)
    
    indices_dict = []
    ## SPLITTING X AND Y, GETTING THE INDEXES
    for train_index, test_index in skf.split(x, y):
        np.random.seed(0)
        np.random.shuffle(train_index)
        np.random.seed(0)
        np.random.shuffle(test_index)
        ## DEFINING INSTANCE TRAINING AND TEST
        names_train_test = {
                'train_names': names[train_index],
                'test_names': names[test_index],
                'train_index' : train_index,
                'test_index': test_index
                }
        indices_dict.append(names_train_test)
        if verbose is True:
            print("Splitting training and testing index")
            print("TRAIN:", train_index, "TEST:", test_index)
            print("TRAIN NAME:", names_train_test['train_names'])
            print("TEST NAME:", names_train_test['test_names'])
            
    return indices_dict
    

### FUNCTION TO AUGMENT DATA
def augment_data(x_train, y_train, data_shape = (32, 32, 32, 3)):
    '''
    The purpose of this function is to augment the training data. It does the following:
        -Rotate training 90 degrees in x- direction
        -Rotate training 90 degrees in y- direction
        -Rotate training 90 degrees in the z direction
        -Contatenate all possible training data
        -Add training Y-values 4x
    INPUTS:
        x_train: [np.array]
            x training data set
        y_train: [np.array]
            y training set
    OUTPUTS:
        x_train: [np.array]
            updated x training data set
        y_train: [np.array]
            updated y training set
        num_tile: [int]
            number of tiles required due to augmentation
    '''
    # 3D ARRAY
    if len(data_shape) == 4:
        ## AUGMENTING X VALUES
        x_train_xy_1 = sp.ndimage.interpolation.rotate(x_train, 90, (1,2))
        x_train_xy_2 = sp.ndimage.interpolation.rotate(x_train, 180, (1,2))
        x_train_xy_3 = sp.ndimage.interpolation.rotate(x_train, 270, (1,2))
        
        x_train_xz_1 = sp.ndimage.interpolation.rotate(x_train, 90, (1,3))
        x_train_xz_2 = sp.ndimage.interpolation.rotate(x_train, 180, (1,3))
        x_train_xz_3 = sp.ndimage.interpolation.rotate(x_train, 270, (1,3))
        
        x_train_yz_1 = sp.ndimage.interpolation.rotate(x_train, 90, (2,3))
        x_train_yz_2 = sp.ndimage.interpolation.rotate(x_train, 180, (2,3))
        x_train_yz_3 = sp.ndimage.interpolation.rotate(x_train, 270, (2,3))
        ## TRAINING SET CONCATENATION
        x_train = np.concatenate((x_train, 
                                  x_train_xy_1, x_train_xy_2, x_train_xy_3,
                                  x_train_xz_1, x_train_xz_2, x_train_xz_3,
                                  x_train_yz_1, x_train_yz_2, x_train_yz_3,
                                  ), axis=0)
        ## DEFINING NUMBER OF TILE
        num_tile = 10

    # 2D ARRAY
    elif len(data_shape) == 3:
        ## ROTATING 90
        x_train_xy_1 = sp.ndimage.interpolation.rotate(x_train, 90, (1,2))
        ## ROTATING 180
        x_train_xy_2 = sp.ndimage.interpolation.rotate(x_train_xy_1, 90, (1,2))
        ## ROTATING 270
        x_train_xy_3 = sp.ndimage.interpolation.rotate(x_train_xy_2, 90, (1,2))
        ## TRAINING SET CONCATENATION
        x_train = np.concatenate((x_train, 
                                  x_train_xy_1, x_train_xy_2, x_train_xy_3,
                                  ), axis=0)
        ## DEFINING NUMBER OF TILE
        num_tile = 4

    ## AUGMENTING Y VALUES    
    y_train = np.tile( y_train, num_tile )
    
    return x_train, y_train, num_tile

### FUNCTION TO RANDOMIZE TRAINING SET
def shuffle_train_set(x_train, y_train):
    '''
    The purpose of this script is to randomize training set.
    INPUTS:
        x_train: [np.array, shape=(num_instances, ...)]
            x training instances
        y_train: [np.array, shape=(num_instances, ...)]
            y_labels for the training instances
    OUTPUTS:
        x_train_shuffled: [np.array, shape=(num_instances, ...)]
            shuffled x training array
        y_train:_shuffled: [np.array, shape=(num_instances, ...)]
            shuffled x-labels corresponding to the x trained
    '''
    ## INITIATING RANDOM SEED
    np.random.seed(0)
    ## FINDING TOTAL INSTANCES
    total_instances = len(x_train)
    ## CHECKING IF INSTANCES MATCH
    if total_instances != len(y_train):
        print("Error! X and y labels do not match!")
        print("Total x instances: %d"%(total_instances) )
        print("Total y instances: %d"%( len(y_train) ) )
    ## GENERATING ARRAY
    index = np.arange(total_instances)
    ## SHUFFLING INDEX
    np.random.shuffle(index)
    ## DEFINING NEW X TRAINING AND Y TRAINING
    x_train_shuffled = x_train[index]
    y_train_shuffled = y_train[index]
    return x_train_shuffled, y_train_shuffled, index

## SPLITTING THE DATA BETWEEN TRAINING AND TEST SET
def split_train_test_set(sampling_dict, 
                         x_data = None, 
                         y_label = None, 
                         instances = None, 
                         md_descriptor_list = None ):
    '''
    The purpose of this function is to split training and test set information.
    INPUTS:
        x_data: [np.array]
            x data numpy array. If this it not None, it will override the 
            instances function!
        y_label: [np.array]
            y data numpy array. 
        instances: [class]
            instances from 'combine_instances' function. 
        sampling_dict: [dict]
            sampling dictionary
        md_descriptor_list: [list, default=None]
            None if you have no additional descriptors you want to split and train on. 
            Otherwise, this will generate a separate output that would have the correct splitting
    OUTPUTS:
        x_train: [np.array]
            array of x training data
        x_test: [np.array]
            array of x testing data
        y_train: [np.array]
            array of y training data
        y_test: [np.array]
            array of y testing data
        If md_descriptor_list is not None:
            md_descriptor_list_train: [np.array, num_instances]
                descriptor list to train on
            md_descriptor_list_test: [np.array, num_instances]
                descriptor list to test on
    '''
    ## DEFINING SAMPLING DICT NAME
    sampling_dict_name = sampling_dict['name']

    ## DEFINING X AND Y DATA
    if x_data is None or y_label is None:
        x_data = instances.x_data
        y_label = instances.y_label
        ## FINDING TOTAL INSTANCES
        total_instances = len(instances.instance_names)
    else:
        ## COMPUTING TOTAL INSTANCES FROM INPUT
        total_instances = len(x_data)
    
    ## FINDING SPLITTING INFORMATION FOR TRAINING AND TESTING
    if sampling_dict_name == 'strlearn':
        try:
            split_training = sampling_dict['split_training']
            split_percentage = None
        except KeyError:
            print("Since 'split_training' is not defined in sampling dictionary, we look for 'split_percentage' entries.")
            split_percentage = sampling_dict['split_percentage']
            print("Split percentage: %.2f"%(split_percentage) )
            pass
    
    ## CREATING EMPTY LISTS TO STORE TRAINING AND TEST INFORMATION
    x_train, x_test, y_train, y_test = [], [], [], []
    
    ## CREATING IF NOT EMPTY
    if md_descriptor_list is not None:
        md_descriptor_list_train, md_descriptor_list_test = [], []
    
    ## LOOPING THROUGH INSTANCES AND STORE TRAINING OR TEST SET
    for each_instance in range(total_instances):
        ## EXTRACTING CURRENT INSTANCE
        # current_instance = extract_instance_names( instances.instance_names[each_instance] )        
        ## DEFINING CURRENT X AND Y DATA
        current_x_data = x_data[each_instance]
        current_y_data = y_label[each_instance]
        if sampling_dict_name == 'strlearn':
            ## FINDING SPLITTING DETAIL BASED ON CURRENT X DATA
            if split_percentage is not None:
                ## FINDING SPLIT INDEX
                split_training = get_split_index_of_list_based_on_percentage( current_x_data, 
                                                                              split_percentage = split_percentage,
                                                                              )
            ## LOOPING THROUGH EACH INDEX OF THE X-DATA
            x_data_flatten_lower_time = [ each_x_data for x_index, each_x_data in enumerate(current_x_data) if x_index < split_training ]
            x_data_flatten_upper_time = [ each_x_data for x_index, each_x_data in enumerate(current_x_data) if x_index >= split_training ]
        elif sampling_dict_name == 'spec_train_tests_split':
            ## FINDING LENGTH OF THE DATA
            x_train_length = len(current_x_data) 
            ## DEFINING NUMBER OF TEST AND TRAINING
            num_test = sampling_dict['num_testing']
            num_train = sampling_dict['num_training']
            ## FINDING CUTOFF
            test_index_cutoff = x_train_length - num_test
            train_index_cutoff = x_train_length - num_test - num_train
            ## SEEING IF NEGATIVE
            if train_index_cutoff < 0 or test_index_cutoff < 0:
                print("Error! Training/test index is negative!")
                print("Current instance: %s"%( instances.instance_names[each_instance]) )
                print("Desired number of training / testing: %d / %d"%( num_train, num_test  ) )
                print("Total length of current instance: %d"%( x_train_length )  )
                sys.exit(1)
            ## DECIDING ON THE X LOWER AND UPPER (TRAIN / TEST)
            x_data_flatten_lower_time = [ each_x_data for x_index, each_x_data in enumerate(current_x_data) if x_index >= train_index_cutoff and \
                                                                                                               x_index < test_index_cutoff  ]
            x_data_flatten_upper_time = [ each_x_data for x_index, each_x_data in enumerate(current_x_data) if x_index >= test_index_cutoff ]
            
        else:
            print("Error! Sampling dict name for splitting functions is not found!")
            print("Check train_deep_cnn.py -- split_train_test_set function")
            sys.exit(1)
        
        ## DEFINING Y DATA
        y_data_flatten_lower_time = [ current_y_data for each_x in range(len(x_data_flatten_lower_time)) ]
        y_data_flatten_upper_time = [ current_y_data for each_x in range(len(x_data_flatten_upper_time)) ]
        
        ## TRAINING SET INFORMATION
        x_train.extend(x_data_flatten_lower_time)
        y_train.extend(y_data_flatten_lower_time)
                
        ## TEST SET INFORMATION
        x_test.extend(x_data_flatten_upper_time)
        y_test.extend(y_data_flatten_upper_time)
        
        ## DESCRIPTOR INFORMATION
        ## DEFINING DESCRIPTOR DATA
        if md_descriptor_list is not None:
            ## DEFINING CURRENT DECRIPTOR DATA
            current_descriptor_data = md_descriptor_list[each_instance]
            descriptor_flatten_lower_time = [ current_descriptor_data for each_x in range(len(x_data_flatten_lower_time)) ]
            descriptor_flatten_upper_time = [ current_descriptor_data for each_x in range(len(x_data_flatten_upper_time)) ]
            
            ## APPENDING
            md_descriptor_list_train.extend(descriptor_flatten_lower_time)
            md_descriptor_list_test.extend(descriptor_flatten_upper_time)
        
    ## CONVERTING X TRAINING, ETC. TO ARRAY
    x_train = np.asarray(x_train)
    x_test = np.asarray(x_test)
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)
    
    ## SEEING IF DESCRIPTORS ARE DESIRED
    if md_descriptor_list is not None:
        md_descriptor_list_train = np.asarray(md_descriptor_list_train)
        md_descriptor_list_test = np.asarray(md_descriptor_list_test)
        return x_train, x_test, y_train, y_test, md_descriptor_list_train, md_descriptor_list_test
    else:
        return x_train, x_test, y_train, y_test
########################################
### CLASS FUNCTION TO TRAIN DEEP CNN ###
########################################
class train_deep_cnn:
    '''
    The purpose of this function is to train deep cnns
    INPUTS:
        instances: [obj]
            instances object from combine_instances class
        sampling_dict: [dict]
            sampling dictionary of the training and test sets
        cnn_type: [str]
            cnn type that you would like. Available types are:
                solvent_net: our own generated cnn
        cnn_dict: [dict]
            dictionary containing cnn requirements
        md_descriptors: [list, default=[]]
            list of md descriptors to use for the training -- which will be attached onto the end of the network
        retrain: [logical, default = False]
            True if you want to retrain regardless of the saved weights
        verbose: [logical, default = False]
            True if you want to be verbose about deep cnn
        want_training_test_pickle: [logical, default=False]
            True if you want to print out training/test pickle file
        class_file_path: [str]
            class file path to csv file
    OUTPUTS:
        ## TRAINING AND TEST SET INFORMATION
            self.x_train: [list]
                training set instances
            self.x_test: [list]
                test set instances
            self.y_train: [list]
                training set y labels
            self.y_test: [list]
                test set y labels
        ## FILE INFORMATION
            self.output_file_name: [str]
                output file name for the model
            self.output_weight_name: [str]
                output weights name
            self.output_full_path: [str]
                full path to output file
            self.output_pickle_name: [str]
                output file name for the pickle (i.e. training/test set information)
            self.output_pickle_path: [str]
                full path to pickle file
            self.output_full_path_exists: [logical]
                True or false depending if the output full path exists
            self.output_pickle_path_exists: [logical]
                True or false depending if output pickle exists
            want_basic_name: [logical, default = False]
                True if you want simple names like "model.hdf5
            want_augment: [logical, default = True]
                True if you want to augment your data set with rotations of 
                90, 180, and 270 degrees in x, y, z directions. 
            want_shuffle: [logical, default=True]
                True if you want to shuffle the training data
            num_cross_validation_folds: [int, default = 1]
                number of cross validation folds. If 1, then we will not perform 
                any cross validation training. The idea is that we want to use 
                all of our data appropriately, but we do not want to miss 
                data by placing some labels in thevalidation set. Therefore, we could 
                perform k-fold cross validations to train the model. Note that this 
                will significantly increase the amount of computational time given 
                that we will need to run ~500 epochs for each run! 
        ## MODEL INFORMATION
            self.model: [obj]
                stores the model information and compiles
            self.history_history: [obj]
                stores history information for the model (e.g. model validation, etc.)
    FUNCTIONS:
        find_all_path: finds all path (if not available already)
        store_pickle: stores pickle file
        restore_pickle: restores pickle file information
    '''
    ## INITIALIZING
    def __init__(self, 
                 instances,
                 sampling_dict,
                 cnn_type,
                 cnn_dict,
                 want_descriptors = False,
                 retrain=False,
                 output_path = None,
                 verbose = False,
                 want_training_test_pickle=False,
                 want_basic_name=False,
                 want_shuffle = True,
                 want_augment = True,
                 class_file_path = None,
                 num_cross_validation_folds = 1,
                 ):
        ## STORING INPUTS
        self.sampling_dict = sampling_dict
        self.cnn_type = cnn_type
        self.cnn_dict = cnn_dict
        self.output_path = output_path
        self.retrain = retrain
        self.verbose = verbose
        self.want_training_test_pickle = want_training_test_pickle
        self.want_basic_name = want_basic_name
        self.want_shuffle = want_shuffle
        self.want_descriptors = want_descriptors
        self.want_augment = want_augment
        self.num_cross_validation_folds = num_cross_validation_folds
        ## FINDING ALL PATHS
        self.find_all_path()
        
        ## FINDING INPUT DATA SHAPE
        self.input_data_shape = instances.x_data[0][0].shape
                    
        ## PRINTING
        if self.verbose == True:
            print("*** INPUT ARRAY SHAPE: %s"%( str(self.input_data_shape) ))

        ## DEFINING OUTPUT FILE NAME
        if self.want_basic_name is False:
            self.output_file_name = instances.pickle_name + '-' + cnn_type + '_' + str(self.cnn_dict['epochs'])
        else:
            self.output_file_name = "model"
        
        ## DEFINING NAMES FOR WEIGHTS, CHECKPOINT, AND PICKLE
        self.output_weight_name = self.output_file_name + ".hdf5"
        self.output_checkpoint = self.output_file_name + ".chk"
        self.output_pickle_name = self.output_file_name + ".pickle"
        
        ## DEFINING FILE PATH
        self.output_full_path = os.path.join(self.output_path, self.output_weight_name)
        self.output_chk_path = os.path.join(self.output_path, self.output_checkpoint)
        self.output_pickle_path = os.path.join(self.output_path, self.output_pickle_name)
        
        ## SEEING IF FILE EXISTS
        self.output_full_path_exists = os.path.isfile(self.output_full_path)
        self.output_pickle_path_exists = os.path.isfile(self.output_pickle_path)
        
        ## DECIDING IF THE MODEL NEEDS TO BE TRAINED OR NOT
        if self.output_full_path_exists == False or self.output_pickle_path_exists == False or self.retrain == True:
            retrain_model = True
        else:
            ## RETRAIN IS FALSE
            retrain_model = False
            ## RESTORING ALL OTHER PICKLE INFORMATION
            self.restore_pickle()
        
        ## SEEING IF DESCRIPTORS ARE DESIRED
        if self.want_descriptors is True:
            ## DEFINING INPUTS FOR DESCRIPTOR FUNCTION
            inputs_for_descriptor_fxn=INPUTS_FOR_DESCRIPTOR_FXN.copy() # {**INPUTS_FOR_DESCRIPTOR_FXN}
            ## UPDATING INSTANCES
            inputs_for_descriptor_fxn['instance_list'] = instances.instance_names
            ## CORRECTING FOR PATH
            if class_file_path is not None:
                inputs_for_descriptor_fxn['path_csv'] = class_file_path
            ## UPDATING INPUTS
            # inputs_for_descriptor_fxn['col_names'] = self.md_descriptors
            ## DEFINING DESCRIPTOR INPUTS
            descriptor_inputs = get_descriptor_inputs(**inputs_for_descriptor_fxn)        
            ## RETRAINING MODEL
            if retrain_model is False:
                ## RENORMALIZING INPUTS
                renormalized_inputs = self.descriptor_inputs.transform_test_df(df = descriptor_inputs.output_dfs)
                ## CREATING DESCRIPTOR ARRAY
                self.descriptor_inputs_array = np.array(renormalized_inputs[self.descriptor_inputs.col_names])
            else:
                ## DEFINING DESCRIPTOR INPUTS
                self.descriptor_inputs = descriptor_inputs
                ## CREATING DESCRIPTOR ARRAY
                self.descriptor_inputs_array = np.array(self.descriptor_inputs.output_dfs_normalized[self.descriptor_inputs.col_names])
        
        ## SEEING IF YOU WANT CROSS VALIDATION 
        if num_cross_validation_folds > 1:
            self.indices_dict = get_indices_for_k_fold_cross_validation( instances, 
                                                                    n_splits = num_cross_validation_folds,
                                                                    verbose = False)
            ## DEFINING LOGICAL
            self.want_cross_validation = True
        else:
            self.indices_dict = [[]]
            self.want_cross_validation = False
        
        ## CREATING LIST OF MODELS
        self.model_list = []
        
        ## START LOOP HERE
        for kfold_idx, kfold_dict in enumerate(self.indices_dict):
            
            ## DEFININING TRAINING INDEX
            if self.want_cross_validation == True:
                train_index = kfold_dict['train_index']
                test_index = kfold_dict['test_index']
                ## DEFINING NEW PICKLE
                self.output_weight_name = self.output_file_name + "_fold_" + str(kfold_idx) + ".hdf5"
                self.output_checkpoint = self.output_file_name  + "_fold_" + str(kfold_idx)  +".chk"
                self.output_pickle_name = self.output_file_name + "_fold_" + str(kfold_idx) + ".pickle"

                ## PRINTING
                if self.verbose is True:
                    print("-----------------------------")
                    print("K-cross validation index: %d"%(kfold_idx) )
                    print("Training index: %s"%(', '.join([str(each_value) for each_value in train_index ])))
                    print("Testing index: %s"%(', '.join([str(each_value) for each_value in test_index ])))
                    print("-----------------------------")
                ## DEFINING FULL PATH TO PICKLE
                self.output_full_path = os.path.join(self.output_path, self.output_weight_name)
                self.output_chk_path = os.path.join(self.output_path, self.output_checkpoint)
                self.output_pickle_path = os.path.join(self.output_path, self.output_pickle_name)
                ## SEEING IF FILE EXISTS
                self.output_full_path_exists = os.path.isfile(self.output_full_path)
                self.output_pickle_path_exists = os.path.isfile(self.output_pickle_path)
                ## DECIDING IF THE MODEL NEEDS TO BE TRAINED OR NOT
                if self.output_full_path_exists == False or self.output_pickle_path_exists == False or self.retrain == True:
                    retrain_model = True
                else:
                    ## RETRAIN IS FALSE
                    retrain_model = False
            
            ## GENERATING SPLITTING DATA
            if self.output_full_path_exists == False or \
               self.output_pickle_path_exists == False or \
               retrain_model == True or \
               self.want_training_test_pickle == False:
                ## SPLITTING X Y DATA
                if self.want_descriptors is False:
                    if self.want_cross_validation == False:
                        ## NORMAL TRAINING, NO CROSS VALIDATION
                        self.x_train, self.x_test, self.y_train, self.y_test = split_train_test_set( instances = instances, 
                                                                                                     sampling_dict = sampling_dict )
                    else:
                        ## DEFINING X DATA
                        x_data = np.array(instances.x_data)
                        y_label = np.array(instances.y_label)
                        
                        ## GETTING THE TRAINING DATA
                        self.x_train, _, self.y_train, _ = split_train_test_set( x_data = x_data[train_index], 
                                                                                 y_label = y_label[train_index],
                                                                                 sampling_dict = sampling_dict )
                        
                        ## GETTING THE VALIDATION DATA
                        self.x_test, _, self.y_test, _ = split_train_test_set( x_data = x_data[test_index], 
                                                                                 y_label = y_label[test_index],
                                                                                 sampling_dict = sampling_dict )
                else:
                    ## NEED TO FIX THIS IF YOU WANT CROSS VALIDATION ENABLED
                    self.x_train, self.x_test, self.y_train, self.y_test, self.md_descriptor_list_train, self.md_descriptor_list_test = \
                                                                                                 split_train_test_set( instances = instances, 
                                                                                                                       sampling_dict = sampling_dict,
                                                                                                                       md_descriptor_list = self.descriptor_inputs_array )
                    
                ## RANDOMIZING TRAINING SET
                if self.want_shuffle is True:
                    self.x_train, self.y_train, self.shuffle_index =  shuffle_train_set(x_train = self.x_train,
                                                                                        y_train = self.y_train,
                                                                                        )
                    ## SHUFFLING DESCRIPTORS
                    if self.want_descriptors is True:
                        self.md_descriptor_list_train = self.md_descriptor_list_train[self.shuffle_index]
                
                ## AUGMENT DATA
                if self.want_augment is True:
                    self.x_train, self.y_train, self.num_tile = augment_data( x_train = self.x_train, 
                                                                              y_train = self.y_train,
                                                                              data_shape = self.input_data_shape)
                ## TILING DESCRIPTORS
                if self.want_descriptors is True:
                    self.md_descriptor_list_train = np.tile( self.md_descriptor_list_train, (self.num_tile,1))
    
            ## RETRAINING THE MODEL
            if retrain_model is True:
                ## PRINTING
                if self.verbose == True:
                    ## PRINTING
                    print("---------------------------------------------------------------")
                    if self.output_full_path_exists == False:
                        print("Weights file does not exist: %s"%( self.output_full_path)   )
                    if self.output_pickle_path_exists == False:
                        print("Pickle file does not exist: %s"%(self.output_pickle_path) )
                    print("Retraining with the parameters:")
                    print("   Network: %s"%(self.cnn_type) )
                    print("   Num epochs: %s"%(self.cnn_dict['epochs']) )
                    print("   Validation split: %s"%(self.cnn_dict['validation_split']) )
                    print("   Batch size: %d"%( self.cnn_dict['batch_size'] ))
                    print("---------------------------------------------------------------")
                
                ## SEEING IF YOU WANT DESCRIPTORS
                if self.want_descriptors is True:
                    print("MD descriptor approach is turned on!")
                    print("Adding descriptors to the last layer")
                    print("Descriptor list: %s"%(', '.join(inputs_for_descriptor_fxn['col_names'])) )
                    ## TURN OFF REGRESSION
                    regress = False
                    ## CREATING MODEL FOR DESCRIPTOR
                    descriptor_model = md_descriptor_network(dim = len(self.descriptor_inputs.col_names), regress = regress )
                else:
                    print("No additional descriptors added, regression is used as last layer")
                    regress = True
                    
                ## SELECTING THE MODEL
                if cnn_type == 'solvent_net':
                    cnn_model = solvent_net(input_data_shape = self.input_data_shape, regress = regress)
                elif cnn_type == 'voxnet':
                    cnn_model = voxnet(input_data_shape = self.input_data_shape, regress = regress)
                elif cnn_type == 'orion':
                    cnn_model = orion(input_data_shape = self.input_data_shape, regress = regress)
                elif cnn_type == 'vgg16':
                    # from keras.applications import vgg16
                    # self.model = vgg16.VGG16(weights=None, include_top=True, input_shape=self.input_data_shape)
                    cnn_model = vgg16( input_data_shape = self.input_data_shape )
                    
                ## SEEING IF DESCRIPTORS IS TRUE
                if self.want_descriptors is False:
                    ## RENAMING MODEL
                    self.model = cnn_model
                    ## DEFINING INPUTS
                    self.deep_network_inputs = self.x_train
                else:
                    ## COMBINING APPRAOCHES
                    combined_input = concatenate([cnn_model.output, descriptor_model.output])  # , axis = 1 , axis = 1
                    
                    # our final FC layer head will have two dense layers, the final one
                    # being our regression head
                    # x = Dense(4, activation="relu")(combinedInput)
                    ## ADDING A RELU LAYER
                    x = Dense(4, activation="relu")(combined_input)
                    x = Dense(1, activation="linear")(x)
                    # our final model will accept categorical/numerical data on the MLP
                    # input and images on the CNN input, outputting a single value (the
                    # predicted price of the house)
                    self.model = Model(inputs=[cnn_model.input, descriptor_model.input], outputs=x)
                    
                    ## DEFINING INPUTS
                    self.deep_network_inputs = [ self.x_train, self.md_descriptor_list_train ]
                
                ## COMPILING DETAILS
                self.model.compile(loss=mean_squared_error, optimizer=Adam(lr=0.00001), metrics=self.cnn_dict['metrics'])
                
                ## DEFINING VALIDATION DATA
                if self.want_cross_validation == True:
                    ## DEFINING VALIDATION DATA
                    validation_data = (self.x_test, self.y_test)
                    validation_split= None
                else:
                    validation_data = None
                    validation_split=self.cnn_dict['validation_split']
                
                ## DEFINING CHECKPOINT
                checkpoint = ModelCheckpoint(self.output_chk_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
                callbacks_list = [checkpoint]
                
                ## STORING TIME
                self.time_total = time.time()
                
                ## TRAINING THE MODEL
                self.history = self.model.fit(  x=self.deep_network_inputs, 
                                                y=self.y_train, 
                                                batch_size=self.cnn_dict['batch_size'], 
                                                epochs=self.cnn_dict['epochs'], 
                                                validation_split=validation_split,
                                                shuffle = self.cnn_dict['shuffle'],
                                                validation_data = validation_data,
                                                callbacks=callbacks_list)
                ## STORING HISTORY
                self.history_history = self.history.history
                
                ## MAKING PREDICTIONS ON VALIDATION SET
                if self.want_cross_validation == True:
                    self.y_pred = self.model.predict(self.x_test)
                
                ## STORING THE MODEL
                # Reference: https://keras.io/getting-started/faq/#savingloading-whole-models-architecture-weights-optimizer-state
                self.model.save( self.output_full_path )            
                ## STORING TIME
                self.time_total = time.time() - self.time_total
                ## CONVERTING TIME
                self.time_hms = convertSeconds(self.time_total)   
                ## AFTER TRAINING THE DATA, STORE EVERYTHING
                self.store_pickle()
                ## PRINTING TIME
                print("TOTAL TRAINING TIME:  %d hours, %d minutes, %d seconds ---" % ( tuple(self.time_hms)) )
        
            ## IF YOU DO NOT NEED TO RETRAIN THE MODEL
            else:
                #   if retrain_model is False:
                ## PRINTING
                if self.verbose == True:
                    print("---------------------------------------------------------------")
                    print("Since weights and pickle are found, we are restoring the weights!")
                    print("Weight file: %s"%( self.output_full_path)   )
                    print("Pickle file: %s"%(self.output_pickle_path) )
                    print("---------------------------------------------------------------")
                
                ## RESTORING WEIGHTS
                self.model = load_model(self.output_full_path)
                
                ## PRINTING SUMMARY
                if self.verbose == True:
                    print("--------- SUMMARY ---------")
                    self.model.summary()
                    
                ## STORING
                self.model_list.append(self.model)
                
        return
                                                                                             
    ### FUNCTION TO FIND PATHS
    def find_all_path(self):
        ''' This function looks for all paths to databases, etc. '''
        ## FINDING PATHS
        path_dict = find_paths()
        ## DEFINING PATH TO OUTPUT
        if self.output_path == None:
            self.output_path = path_dict['output_path']
        return
    
    ### FUNCTION TO STORE PICKLE
    def store_pickle(self):
        ''' This function stores the pickle'''
        # FNULL = open(os.devnull, 'r')
        # import subprocess
        # subprocess.Popen( self.output_pickle_path_exists,stdin=FNULL)
        # subprocess.Popen( args=[self.output_pickle_path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        
        ## FINDING TRAINING TEST PICKLE
        if self.want_training_test_pickle == True:
            pickle_dump_list = [self.x_train, self.y_train, self.x_test, self.y_test, self.history_history, self.time_hms]
        elif self.want_cross_validation == True:
            pickle_dump_list = [self.history_history, self.time_hms, self.y_pred, self.y_test, self.indices_dict]
        else:
            pickle_dump_list = [self.history_history, self.time_hms]
        
        ## STORING DESCRIPTOR INPUTS
        if self.want_descriptors is True:
            pickle_dump_list.append(self.descriptor_inputs)
        ## LOADING PICKLE FILE
        with open(self.output_pickle_path, 'wb') as f:  # Python 3: open(..., 'wb')
            ## STORING
            pickle.dump(pickle_dump_list, f, protocol=2)  # <-- protocol 2 required for python2   # -1
        return
    ### RETRIVE PICKLE
    def restore_pickle(self):
        ''' This function restores the pickle '''
        if self.want_training_test_pickle == True:
            if self.want_descriptors is True:
                self.x_train, self.y_train, self.x_test, self.y_test, self.history_history, self.time_hms, self.descriptor_inputs = load_pickle_general(self.output_pickle_path)
            else:
                self.x_train, self.y_train, self.x_test, self.y_test, self.history_history, self.time_hms = load_pickle_general(self.output_pickle_path)
        elif self.want_cross_validation == True:
            self.history_history, self.time_hms, self.y_pred, self.y_test, self.indices_dict = load_pickle_general(self.output_pickle_path)
        else:
            if self.want_descriptors is True:
                self.history_history, self.time_hms, self.descriptor_inputs = load_pickle_general(self.output_pickle_path)
            else:
                self.history_history, self.time_hms = load_pickle_general(self.output_pickle_path)
            


##%%
## MAIN FUNCTION
if __name__ == "__main__":
    
    ## INCLUDING RANDOM SEED
    np.random.seed(0)
    
    ## SEE IF TESTING IS ON
    testing = check_testing()
    
    ## DEFAULT INPUTS
    ## DEFINING NUMBER OF EPOCHS
    cnn_dict = {
            'validation_split': 0.2,
            'batch_size': 18, # higher batches results in faster training, lower batches can converge faster
            'metrics': ['mean_squared_error'],
            'shuffle': True, # True if you want to shuffle the training data
            }
    
#    ## DEFINING SAMPLING INFORMATION
    sampling_dict = {
            'name': 'strlearn',
            'split_percentage': 0.75, # 3,
            }
    
    ## DEFINING SAMPLING INFORMATION
#    sampling_dict = {
#            'name': 'spec_train_tests_split',
#            'num_training': 3, # 3,
#            'num_testing' : 2,
#            }
    
    
    ## SEEING IF TESTING IS TRUE
    if testing == True:
    
        ## DEFINING SOLVENT LIST
        solvent_list = [ 'DIO' ]# 'GVL', 'THF' ] # , 'GVL', 'THF'  , 'GVL', 'THF'
        ## DEFINING MASS FRACTION DATA
        mass_frac_data = ['10'] # , '25', '50', '75' , '25', '50', '75'
        ## DEFINING SOLUTE LIST
        solute_list = ['XYL', 'FRU', "LGA", "ETBE", "PDO"] # list(SOLUTE_TO_TEMP_DICT) , 'FRU'
        ## DEFINING TYPE OF REPRESENTATION
        # representation_type = 'split_avg_nonorm_planar' # split_avg_nonorm, #split_average
#        representation_type = 'split_avg_nonorm_sampling_times' # split_avg_nonorm, #split_average split_avg_nonorm
#        representation_inputs = {
#                'num_splits': 10,
#                'perc': 1.00,
#                'initial_frame': 0,
#                'last_frame': 1000,
#                }
        representation_type = 'split_avg_nonorm' # split_avg_nonorm, #split_average split_avg_nonorm
        representation_inputs = {
                'num_splits': 10,
                }
        
        ## DEFINING DATA TYPE
        # data_type="30_30"
        data_type="20_20_20_20ns_firstwith10_oxy"
        # "16_16_16_32ns_first" # 20_20_20_32ns_first
        # 10_10_10_32ns_first
        # "20_20_20_32ns_first"
        # "20_20_20" 
        # "30_30_x30"
        
        ## DEFINING PATHS
        database_path = r"R:\scratch\SideProjectHuber\Analysis\CS760_Database_20_20_20" # None # Since None, we will find them!
        database_path = os.path.join( r"R:\scratch\3d_cnn_project\database", data_type)  # None # Since None, we will find them!        
        class_file_path = r"R:\scratch\3d_cnn_project\database\Experimental_Data\solvent_effects_regression_data.csv"
        # None
        combined_database_path = r"R:\scratch\3d_cnn_project\combined_data_set"
        output_file_path = r"R:\scratch\3d_cnn_project\simulations" # OUTPUT PATH FOR CNN NETWORKS
        
        ## DEFINING VERBOSITY
        verbose = True

        ## SELECTING TYPE OF CNN
        cnn_type = 'voxnet' # 'orion' # 'solvent_net' 'voxnet'
        '''
        voxnet
        orion
        solvent_net
        '''
        ## DEFINING NUMBER OF EPOCHS
        cnn_dict['epochs'] = 2
        # 1
        
        ## DEFINING RETRAINING DETAILS
        retrain = True # False
        
        ## LOGICAL FOR STORING AND RELOADING TEST PICKLES
        want_training_test_pickle = False
        want_basic_name = False
        want_descriptors = False
        want_shuffle = False
        want_augment= True
        
        ## DEFINING NUMBER OF CROSS VALIDATION FOLDING
        num_cross_validation_folds = 5
        # 1
        # 1 for no cross validations
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
        parser.add_option("-v", dest="verbose", action="store_true", )
        
        ## DEFINING DATA SET TYPE
        parser.add_option('-z', '--datatype', dest = 'data_type', help = 'data type', type="string", default = "20_20_20")
        
        ## DEFINING RETRAINING IF NECESSARY
        parser.add_option("-t", dest="retrain", action="store_true", default = False )
        
        ## STORING TRAINING PICKLE
        parser.add_option("-p", dest="want_training_test_pickle", action="store_true", default = False )

        ## AUGMENTATION
        parser.add_option("--no_augment", dest="want_augment", action="store_false", default = True ) 
        
        ## CROSS VALIDATION FOLDS
        parser.add_option("--num_cross_folds", dest="num_cross_validation_folds", help = "Number of cross validation folds", default = 1 ) 
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
        
        ## DEFINING DATA TYPE
        data_type = options.data_type
        
        ## FILE PATHS
        database_path = options.database_path
        class_file_path = options.class_file_path
        combined_database_path = options.combined_database_path
        output_file_path = options.output_file_path
        
        ## SEEING IF FILE PATHS
        if database_path == "None":
            database_path = None
        if class_file_path == "None":
            class_file_path = None
        if combined_database_path == "None":
            combined_database_path = None
        if output_file_path == "None":
            output_file_path = None
        
        ## VERBOSITY
        verbose = options.verbose
        
        ## RETRAIN
        retrain = options.retrain
        
        ## TRAINING TEST PICKLE
        want_training_test_pickle = options.want_training_test_pickle
        want_augment = options.want_augment

        ## ADDING EPOCHS
        cnn_dict['epochs'] = options.epochs
        
        ## UPDATING REPRESATION INPUTS
        representation_inputs = extract_representation_inputs( representation_type = representation_type, 
                                                              representation_inputs = representation_inputs )
        
        ## DEFINING CROSS VALIDATION FOLDS
        num_cross_validation_folds = options.num_cross_validation_folds
        
            
        
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

    #%%
    
    ### TRAINING CNN
    deep_cnn = train_deep_cnn(
                     instances = instances,
                     sampling_dict = sampling_dict,
                     cnn_type = cnn_type,
                     cnn_dict = cnn_dict,
                     retrain=retrain,
                     output_path = output_file_path,
                     verbose = verbose,
                     want_training_test_pickle = want_training_test_pickle,
                     want_basic_name = want_basic_name,
                     want_descriptors = want_descriptors,
                     want_augment = want_augment,
                     num_cross_validation_folds = num_cross_validation_folds,
                     )

    
    #%%
    


#    ## GETTING INDICES
#    indices_dict = get_indices_for_k_fold_cross_validation( instances, 
#                                                            n_splits = 5,
#                                                            verbose = True)

    #%%
    
    
    '''
    ## TODO: check if this is working correctly (rotation)
    x_train = deep_cnn.x_train
    x_train_xy_1 = sp.ndimage.interpolation.rotate(x_train, 0, (1,2))
    import tensorflow as tf
    x_train_xy_1 = tf.image.resize_image_with_crop_or_pad(
            image = x_train_xy_1,
            target_height = 32,
            target_width = 32
        )
#    x_train_xy_1 = sp.ndimage.interpolation.rotate(x_train_xy_1, 90, (1,2))
#    x_train_xy_1 = sp.ndimage.interpolation.rotate(x_train_xy_1, 90, (1,2))
#    x_train_xy_1 = sp.ndimage.interpolation.rotate(x_train_xy_1, 90, (1,2))
    # x_train_xy_1 = rotate(x_train, 0, (2,1) ) # (1,2)
    
    # x_train_xy_1 = rotate(x_train[0],180,axes = (0,1), reshape=False)
    
    ## AVERAGING ALONG THE X-DIRECTION
#    instances_x_avged = x_train[0] # np.mean(instances.x_data[0][0],axis=2)
    instances_x_avged = x_train_xy_1[0] # np.mean(instances.x_data[0][0],axis=2)
    # instances_x_avged = x_train_xy_1 # np.mean(instances.x_data[0][0],axis=2)
    
    #--- VISUALIZING
    ## IMPORTING MODULES
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
    import numpy as np
    from core.plotting_scripts import renormalize_rgb_array
    
    instances_x_avged_norm =  renormalize_rgb_array(instances_x_avged) # instances_x_avged #
    # instances_x_avged_norm =  instances_x_avged
    
    ## PLOTTING
    fig, ax = plt.subplots()
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    
    # ax.imshow(X = instances_x_avged_norm, alpha = 0.5, vmax = abs(instances_x_avged_norm).max(), vmin=abs(instances_x_avged_norm).min())
    ax.imshow(X = instances_x_avged_norm, alpha = 1) # interpolation='bilinear', 
    # ax.imshow(X = instances_x_avged_norm, alpha = 1)
    
    '''
    
    #%%
    '''
    ############################
    ##### ACCURACY METRICS #####
    ############################
    
    ## IMPORTING SKLEARN MODULES
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import explained_variance_score
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import r2_score
    
    ## COMPUTING ROOT MEAN SQUARED ERROR
    def compute_rmse(predictions, targets):
        return np.sqrt(((predictions - targets) ** 2).mean())
    
    ## FUNCTION TO CALCUALTE MSE, R2, EVS, MAE
    def metrics(y_fit,y_act):
        evs = explained_variance_score(y_act, y_fit)
        mae = mean_absolute_error(y_act, y_fit)
        rmse = compute_rmse(predictions=y_fit, targets = y_act) # mean_squared_error(y_act, y_fit)
        r2 = r2_score(y_act, y_fit)
        return mae, rmse, evs, r2
    
    ## MAKING PREDICTIONS
    model = deep_cnn.model
    x_test = deep_cnn.x_test
    
    ## PREDICTIONS
    y_pred = model.predict(x_test).reshape(len(deep_cnn.y_test) )
    y_true = deep_cnn.y_test
    
    ## COMPUTING ACCURACY (MEAN AVERAGE ERROR, ROOT MEAN SQUARED ERROR, etc.)
    mae, rmse, evs, r2 = metrics(y_fit = y_pred, 
                                y_act = y_true)
    
    
    #%%
    ########################
    ##### PARITY PLOT ######
    ########################
    import matplotlib.pyplot as plt
    from core.plotting_scripts import DEFAULT_FIG_INFO, LABELS_DICT, LINE_STYLE, update_ax_limits, AXIS_RANGES, SAVE_FIG_INFO, \
                                         change_axis_label_fonts, TICKS_DICT, get_cmap
    
    ### PLOT PARITY PLOT
    def plot_parity( true_values,
                     pred_values,
                     ax = None, 
                     fig = None, 
                     title = None,
                     loc = 'lower right',
                     save_fig = False,
                     fig_name="parity_plot"
                     ):
        ## CHECKING IF FIGURE IS CREATED
        if ax == None or fig == None:
            fig = plt.figure(**DEFAULT_FIG_INFO) 
            ax = fig.add_subplot(111)
            ## DRAWING LABELS
            if title is not None:
                ax.set_title( title ,**LABELS_DICT)
            ax.set_xlabel('Actual values',**LABELS_DICT)
            ax.set_ylabel('Predicted values',**LABELS_DICT)
        

        ## PLOTTING
        ax.scatter( true_values, pred_values, marker = 'o', color='k', linewidth=.1 ) # label = "Max tree depth: %s"%(max_depth)

        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
            np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
        ]
        ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
        ax.set_aspect('equal')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        
        ## SETTING X TICKS AND Y TICKS
        ax.set_xticks(np.arange(lims[0], lims[1], 1))
        ax.set_yticks(np.arange(lims[0], lims[1], 1))

        ## TIGHT LAYOUT
        fig.tight_layout()

        ## SETTING EQUAL AXIS
        # ax.axis('equal')
        # fig.show()
        # ax.set_aspect('equal', adjustable='box')
        return fig, ax
    
    fig, ax = plot_parity( true_values = y_true,
                           pred_values = y_pred, 
                          )
    # '''
    
    
 