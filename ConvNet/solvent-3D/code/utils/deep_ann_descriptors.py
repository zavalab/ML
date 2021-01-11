# -*- coding: utf-8 -*-
"""
deep_ann_descriptors.py
The purpose of this script is to run a artifical neural network when using 
the 3 descriptor framework

Created on: 07/01/2019

Author(s):
    - Alex K. Chew (alexkchew@gmail.com)
    
References:
    Compile your first neural network: https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
"""

## IMPORTING MODULES
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import time
import os
import pickle

## KERAS MODELS
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.losses import mean_squared_error
from keras.models import load_model

## TIME
from core.track_time import convert_sec_to_hms

## CUSTOM MD DESCRIPTOR ANALYSIS
from extract_md_descriptors import analyze_descriptor_approach, leave_one_out_cross_validation_training_testing

## SHUFFLING TOOLS
from train_deep_cnn import shuffle_train_set

## ANALYSIS TOOLS
from analyze_deep_cnn import metrics, plot_learning_curve

## IMPORTING GLOBAL VARS
from core.global_vars import CNN_DICT

## TAKING EXTRACTION SCRIPTS
from extraction_scripts import load_pickle_general

## PLOTTING
from read_extract_deep_cnn import plot_parity_publication_single_solvent_system

## GETTING PEARSON'S R
from scipy.stats import pearsonr

## FUNCTION TO NORMALIZE AND SHUFFLE DATA
def normalize_and_shuffle_data(
                         input_array,
                         output_array,
                         want_shuffle = True):
    '''
    This function simply loads the descriptor data.
    INPUTS:
        input_array: [np.array]
            input array to be normalized
        output_array: [np.array]
            output array that can be reshuffled
        want_shuffle: [logical, default=True]
            True if you want to shuffle the training set
    OUTPUTS:
        min_max_scalar: [object]
            min max scaler
                to transform: min_max_scalar.fit_transform( np.array(input_array) ) 
                to use current fits: 
        input_array_normalized_shuffled: [np.array]
            input array normalized 
        output_array_shuffled: [array]
            numpy array of the outputs
        shuffle_index: [np.array]
            shuffled index
    '''
    ## NORMALIZING INPUTS
    min_max_scalar = MinMaxScaler()
    input_array_normalized = min_max_scalar.fit_transform( np.array(input_array) ) 
    ## SHUFFLING INPUTS
    if want_shuffle is True:
        ## PRINTING
        print("Shuffling the training set!")
        ## SHUFFLING
        input_array_normalized_shuffled, output_array_shuffled, shuffle_index = shuffle_train_set(x_train = input_array_normalized,
                                                                                                  y_train = output_array)
    else:
        print("Training set is not shuffled!")
        input_array_normalized_shuffled = np.copy( input_array_normalized ) 
        output_array_shuffled = np.copy( output_array )
        shuffle_index = np.arange(len(input_array_normalized_shuffled))
    return input_array_normalized_shuffled, output_array_shuffled, shuffle_index, min_max_scalar

### DEFINING NETWORK
def mlp_network(input_dim, neuron_per_layer = [6], activation = "relu", regress = True ):
    '''
    The purpose of this function is to develop a multi-layer perceptron network using TensorFlow. 
    Here, we vary the number of available neurons and the number of dense layers. 
    The neurons per dense layers is listed as a list.
    Reference: https://www.pyimagesearch.com/2019/02/04/keras-multiple-inputs-and-mixed-data/
    INPUTS:
        input_dim: [int]
            input dimension of your data set. e.g. 3 for 3 parameters
        neuron_per_layer: [list]
            number of neurons per layer
        activation: [str, default='relu']
            activation for each neuron
        regress: [logical, default=True]
            true if you want regression task
    '''
    ## DEFINING THE MODEL
    model = Sequential()
    ## LOOPING TO CREATE NEW LAYERS
    for idx, num_neuron_per_layer in enumerate(neuron_per_layer):
        if idx == 0:
            model.add(Dense(num_neuron_per_layer, input_dim=input_dim, activation=activation))
        else:
            ## JUST ADD TO THE MODEL
            model.add(Dense(num_neuron_per_layer, activation=activation))
 
	## CHECK TO SEE IF REGRESSION IS NEEDED TO BE ADDED
    if regress:
        model.add(Dense(1, activation="linear"))
 
	# return our model
    return model

### FUNCTION TO GET OUTPUT NAME
def get_output_name_descriptors( num_epochs,
                                 neurons_per_layer,
                                ):
    '''
    The purpose of this function is to get the output name for each descriptor. 
    INPUTS:
        num_epochs: [int]
            number of epochs
        neurons_per_layer: [list]
            list of neurons per layer
    OUTPUTS:
        output_name: [str]
            string containing output name
    '''
    ## DEFINING LIST
    output_name_list = [
            str(num_epochs),
            '_'.join( [str(each) for each in neurons_per_layer] )
            ]
    output_name = '-'.join(output_name_list)
    return output_name


### FUNCTION TO EXTRACT INPUT AND OUTPUT ARRAY
def extract_df_input_output_array( df,
                                   input_cols,
                                   output_cols,):
    '''
    This function extracts input and output array from a dataframe.
    INPUTS:
        df: [dataframe]
            dataframe you want to extract details from
        input_cols: [list]
            list of columns you want from dataframe
        output_cols: [str]
            string of output columns
    OUTPUTS:
        input_array: [np.array]
            numpy array of the inputs
        output_array: [np.array]
            numpy array of the outputs
    '''
    input_array = np.array(df[input_cols])
    output_array = np.array(df[output_cols])
    return input_array, output_array
    

##########################################
### CLASS FUNCTION FOR ANN DESCRIPTORS ###
##########################################
class nn_descriptors_model:
    '''
    The purpose of this class is to use neural network on descriptor inputs.
    INPUTS:
        path_md_descriptors: [str]
            str to csv file of md descriptors
        path_sim: [str]
            path to the simulation to store pickles, checkpoints, etc.
        neuron_list: [list]
            list of list that you want to check the neuron list
        nn_dict: [dict]
            dict to run neural network
        learning_rate: [float]
            learning rate for the network
        analyze_descriptor_approach_inputs: [dict]
            dictionary for descriptor approach
    OUTPUTS:
        ## LOADING DESCRIPTOR INFO
            self.analyzed_descriptors: [class object]
                analyzed descriptor
            self.output_array: [array]
                numpy array of the outputs
            self.input_array_normalized: [np.array]
                min max rescaled variables
        ## TRAINING INFO
            self.model_list: [list]
                list of the models with weights
            self.history_list: [list]
                list of history
            self.output_name_list: [list]
                list of all output names
        ## DATAFRAME INFO
            self.predict_df: [list]
                list of dataframes for predicted values. The dataframe could be 
                used for producing parity plots
            self.predict_stats: [list]
                list of predicted states
    '''
    ## INITIALIZING
    def __init__(self,
                 path_md_descriptors,
                 path_sim,
                 neuron_list,
                 analyze_descriptor_approach_inputs,
                 nn_dict,
                 learning_rate = 0.001,
                 ):
        ## STORING
        self.neuron_list = neuron_list
        self.path_md_descriptors = path_md_descriptors
        self.path_sim = path_sim
        self.nn_dict = nn_dict
        self.learning_rate = learning_rate
        self.analyze_descriptor_approach_inputs = analyze_descriptor_approach_inputs
        
        ## LOADING DESCRIPTORS
        self.analyzed_descriptors = analyze_descriptor_approach(**self.analyze_descriptor_approach_inputs)
        
        ## TRAINING ALL MODEL DATA
        self.train_model_all_data()
        
        return
    
    ## FUNCTION TO TRAIN MODEL WITH ALL THE DATA
    def train_model_all_data(self):
        '''
        The purpose of this function is to train all the model data. 
        
        '''
        ## DEFINING INPUTS AND OUTPUTS
        self.input_array, self.output_array = extract_df_input_output_array(
                                                                            df = self.analyzed_descriptors.csv_file,
                                                                            input_cols = self.analyze_descriptor_approach_inputs['molecular_descriptors'],
                                                                            output_cols = self.analyze_descriptor_approach_inputs['output_label'],
                                                                            )
        
        ## SHUFFLING DATA
        self.input_array_normalized_shuffled, self.output_array_shuffled, shuffle_index, self.min_max_scalar=  normalize_and_shuffle_data(
                                                                                                              input_array = self.input_array,
                                                                                                              output_array = self.output_array,                                                                                    
                                                                                                              )
        
        ## TRAINING THE MODEL
        self.model_list, self.history_list, self.output_name_list = self.train_model(input_x_array = self.input_array_normalized_shuffled,
                                                                                     input_y_array = self.output_array_shuffled) # retrain = True
        
        ## GENERATE PREDICTIVE DF
        self.predict_df, self.predict_stats = self.generate_df_predict_training_set(df = self.analyzed_descriptors.csv_file,
                                                                                    input_array_col_names = self.analyze_descriptor_approach_inputs['molecular_descriptors'],
                                                                                    output_array_col_names = self.analyze_descriptor_approach_inputs['output_label'],
                                                                                    min_max_scalar = self.min_max_scalar,
                                                                                    )
        
        
        return
        
    
    ## TRAIN THE MODEL
    def train_model(self, 
                    input_x_array,
                    input_y_array,
                    ending_string='',
                    retrain = False,
                    verbose = True):
        '''
        The purpose of this function is to train the model. 
        INPUTS:
            self: [obj]
                self object
            input_x_array: [np.array]
                input x array to train on
            input_y_array: [np.array]
                input y array to train on
            ending_string: [str, default = '']
                ending string for saving the file
            retrain: [logical, default=False]
                True if you want to retrain the model
            verbose: [logical, default=True]
                True if you want to verbosely print out training the model
        OUTPUTS:
            self.model_list: [list]
                list of the models with weights
            self.history_list: [list]
                list of history
            self.output_name_list: [list]
                list of all output names
        '''
        ## DEFINING A WAY TO LOAD THE MODEL
        model_list = []
        history_list = []
        output_name_list = []
        
        ## LOOPING THROUGH EACH NEURON COMBINATION
        for neurons_per_layer in self.neuron_list:    

            ## DEFINING OUTPUT
            output_name=get_output_name_descriptors( num_epochs = self.nn_dict['epochs'],
                                                     neurons_per_layer = neurons_per_layer, )
            ## DEFINING FILE PATHS
            output_file_path=os.path.join( self.path_sim , output_name + ending_string )
            output_chk_path= output_file_path + '.chk'
            output_weights_path = output_file_path + '.hdf5'
            output_pickle = output_file_path + '.pickle'
            
            ## PRINTING
            if verbose is True:
                print("Working on neurons: %s" %(output_name ))                
                print("Weight path: %s"%( output_weights_path ) )
                print("Pickle path: %s"%( output_pickle ))
            
            ## CHECKING IF THE CHECK POINT AND WEIGHTS ARE EXISTING
            if os.path.isfile(output_weights_path) is False or os.path.isfile(output_pickle) is False or retrain is True:
                ## PRINTING
                if verbose is True:
                    print("Since weight and pickle is not found -- RETRAINING!")
                    print("Retrain: %s"%( retrain ))
                ########################
                ### RETRAINING MODEL ###
                ########################
                ## CREATING THE MODEL
                model = mlp_network(input_dim = np.size(input_x_array, axis=1), 
                                    neuron_per_layer = neurons_per_layer, 
                                    activation = "relu", 
                                    regress = True )
            
                ## COMPILING DETAILS
                model.compile(loss=mean_squared_error, optimizer=Adam(lr=self.learning_rate), metrics=self.nn_dict['metrics'])
                checkpoint = ModelCheckpoint(output_chk_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
                callbacks_list = [checkpoint]
                
                ## STORING TIME
                time_total = time.time()
                
                ## TRAINING THE MODEL
                history = model.fit(x=input_x_array, 
                                    y=input_y_array, 
                                    batch_size=self.nn_dict['batch_size'], 
                                    epochs=self.nn_dict['epochs'], 
                                    validation_split=self.nn_dict['validation_split'], 
                                    shuffle = self.nn_dict['shuffle'],
                                    callbacks=callbacks_list)
                ## SAVING MODEL
                model.save( output_weights_path )
            
                ## STORING TIME
                time_total = time.time() - time_total
                ## CONVERTING TIME
                h,m,s = convert_sec_to_hms(time_total)
                
                ## PRINTING
                print("Total training time is: %d hrs, %d mins, %d sec"%(h, m, s) )
                
                ## STORING
                pickle_dump_list = [history, time_total]
                
                ## LOADING PICKLE
                with open(output_pickle, 'wb') as f:  # Python 3: open(..., 'wb')
                    ## STORING
                    pickle.dump(pickle_dump_list, f, protocol=2)  # <-- protocol 2 required for python2   #
            
            else:
                ## PRINTING
                if verbose is True:
                    print("Since training is found, RELOADING...")
                ###########################
                ### RELOADING THE MODEL ###
                ###########################
                ## LOADING MODEL
                model = load_model(output_weights_path)
                ## LOADING PICKLE
                history, time_total = load_pickle_general(output_pickle)
                
            ## STORING THE MODEL
            model_list.append(model)
            history_list.append(history)
            output_name_list.append(output_name)
            
        return model_list, history_list, output_name_list
    
    ### FUNCTION TO TRAIN SPECIFIC DATA
    def train_across_data(self,
                          column_name = 'cosolvent',
                          suffix_name = '-specific_data',):
        '''
        The purpose of this function is to train the neural network across data, e.g. cosolvents, or reactants, and so on.
        INPUTS:
            column_name: [str]
                column name of data that you are interested in
            suffix_name: [str]
                suffix name of the data you would like
        OUTPUTS:
            
        '''
        ## DEFINING CSV FILE
        csv_file = self.analyzed_descriptors.csv_file
        
        ## FINDING ALL UNIQUE COLUMNS
        unique_columns = np.unique(csv_file[column_name])
        
        ## DEFINING STORAGE
        model_storage = {}
        
        ## LOOPING THROUGH EACH UNIQUE COLUMN
        for current_column in unique_columns:
            ## PRINTING
            print("Training across %s : %s"%(column_name, current_column) )
            ## DEFINING DATASET
            train_df = csv_file[csv_file[column_name] == current_column]
            
            ## DEFINING NAME
            training_name = suffix_name + '-' + column_name + '-' + current_column
            
            ## DEFINING INPUT AND OUTPUT ARRAY
            input_array, output_array = extract_df_input_output_array(
                                                                    df = train_df,
                                                                    input_cols = self.analyze_descriptor_approach_inputs['molecular_descriptors'],
                                                                    output_cols = self.analyze_descriptor_approach_inputs['output_label'],
                                                                    )
            
            ## SHUFFLING TRAINING DATA
            input_array_normalized_shuffled, output_array_shuffled, shuffle_index, min_max_scalar = normalize_and_shuffle_data(
                                                                                                                  input_array = input_array,
                                                                                                                  output_array = output_array,                                                                                    
                                                                                                                  )
            
            ## TRAINING THE MODEL
            model_list, history_list, output_name_list = self.train_model(input_x_array = input_array_normalized_shuffled,
                                                                          input_y_array = output_array_shuffled,
                                                                          ending_string = training_name) # retrain = True
            
            ## GENERATE PREDICTIVE DF
            predict_df, predict_stats = self.generate_df_predict_training_set(df = train_df,
                                                                              input_array_col_names = self.analyze_descriptor_approach_inputs['molecular_descriptors'],
                                                                              output_array_col_names = self.analyze_descriptor_approach_inputs['output_label'],
                                                                              min_max_scalar = min_max_scalar,
                                                                              )
            model_storage[current_column] = {
                    'predict_df': predict_df,
                    'predict_stats': predict_stats,
                    }
            
        return model_storage
        
        
    
    ### FUNCTION TO GET DATABASE AND PREDICT FOR EACH NEURON
    def generate_df_predict_training_set(self, 
                                         df,
                                         input_array_col_names = ['gamma', 'tau', 'delta'], 
                                         output_array_col_names = 'sigma_label',
                                         min_max_scalar = None,
                                         verbose = False):
        '''
        The purpose of this function is to generate predicted training set 
        for each neuron. 
        INPUTS:
            self: [obj]
                self object
            df: [dataframe]
                dataframe containing all input information
            input_array_col_names: [list]
                list of columns that you plan to use
            output_array_col_names: [str]
                output name of the column
            min_max_scalar: [obj, default=None]
                re-scales input array
            verbose: [logical, default=False]
                True if you want to print out details
        OUTPUTS:
            predict_df: [list]
                list of dataframes for predicted values. The dataframe could be 
                used for producing parity plots
            predict_stats: [list]
                list of predicted states
        '''
        ## CREATING LIST
        predict_df = []
        predict_stats = []
        
        ## DEFINING INPUT AND OUTPUT ARRAY
        input_array, output_array = extract_df_input_output_array(
                                                                df = df,
                                                                input_cols = input_array_col_names,
                                                                output_cols = output_array_col_names,
                                                                )
        
        ## PRINTING
        if verbose is True:
            print("--- Predicting values with the model ---")
        
        ## RENORMALIZING THE INPUT ARRAY
        if min_max_scalar is not None:
            if verbose is True:
                print("Renormalizing the input array!")
            ## NORMALIZING
            input_array_normalized = min_max_scalar.transform( np.array(input_array) )
        else:
            ## PRINTING
            if verbose is True:
                print("Since no min-max scalar inputted, no renormalization was performed!")
            ## REDEFINING VARIABLE
            input_array_normalized = np.array(input_array)
            
        ## LOOPING THROUGH EACH
        for idx, neurons_per_layer in enumerate(self.neuron_list):
            ## DEFINING THE MODEL
            model = self.model_list[idx]
            ## PREDICTING SIGMA WITH TRAINING SET
            y_pred = model.predict( input_array_normalized ).reshape(len(output_array) )
            ## ANALYZING THE RESULTS
            mae, rmse, evs, r2, slope = metrics(y_fit = y_pred, 
                                                y_act = output_array)
            ## COMPUTING PEARSONS R
            pearson_r = pearsonr( x = output_array, y = y_pred )[0]
            
            ## CREATING DATAFRAME
            df_pred = df.copy()
            
            ## ADDING TO DATAFRAME
            df_pred['y_pred'] = y_pred[:]
            ## APPENDING
            predict_df.append(df_pred)
            ## APPENDING
            predict_stats.append(
                    {
                            'mae': mae,
                            'rmse': rmse,
                            'evs': evs,
                            'r2': r2,
                            'slope': slope,
                            'pearson_r' : pearson_r, 
                            }
                    )
        return predict_df, predict_stats
    
    ### FUNCTION TO CROSS VALIDATION
    def cross_validate(self,
                       column_name = 'cosolvent',
                       verbose = True,
                       ):
        '''
        The purpose of this function is to cross validate across different column names. 
        This function is run after cross validation across all systems.
        INPUTS:
            df: [dataframe]
                dataframe containg all information
            column_name: [str]
                name of the column you want to cross validate
            verbose: [logical, default=True]
                True if you want to print out all details
        OUTPUTS:
            
        '''
        ## DEFINING CSV FILE
        csv_file = self.analyzed_descriptors.csv_file
        
        ## FINDING ALL UNIQUE COLUMNS
        unique_columns = np.unique(csv_file[column_name])
        ## GENERATE CROSS VALIDATION LIST
        cross_validation_list = leave_one_out_cross_validation_training_testing(data = unique_columns.tolist())
        ## DEFINING STORAGE
        model_storage = {}
        
        ## LOOPING THROUGH CROSS VALIDATION LIST
        for each_cross_data in cross_validation_list:

            ## DEFININING TRAINING/TEST SET
            train_set = each_cross_data[0]
            test_set = each_cross_data[1]
            
            ## PRINTING
            if verbose is True:
                print("\n--- Cross-validating across %s: %s ---"%(column_name, test_set))
            
            ## DEFINING CURRENT NAME
            cross_valid_name = '-cross_valid-' + column_name + '-' +  test_set
            
            ## DEFINING DATAFRAMES FOR TRAINING AND TEST SET
            train_df = csv_file[csv_file[column_name].isin(train_set)]
            test_df = csv_file[csv_file[column_name].isin([test_set])]
            
            ## DEFINING INPUT AND OUTPUT ARRAY
            input_array, output_array = extract_df_input_output_array(
                                                                    df = train_df,
                                                                    input_cols = self.analyze_descriptor_approach_inputs['molecular_descriptors'],
                                                                    output_cols = self.analyze_descriptor_approach_inputs['output_label'],
                                                                    )
            ## SHUFFLING TRAINING DATA
            input_array_normalized_shuffled, output_array_shuffled, shuffle_index, min_max_scalar = normalize_and_shuffle_data(
                                                                                                                  input_array = input_array,
                                                                                                                  output_array = output_array,                                                                                    
                                                                                                                  )
            
            ## TRAINING THE MODEL
            model_list, history_list, output_name_list = self.train_model(input_x_array = input_array_normalized_shuffled,
                                                                          input_y_array = output_array_shuffled,
                                                                          ending_string = cross_valid_name) # retrain = True
            
            ## GENERATE PREDICTIVE DF
            predict_df, predict_stats = self.generate_df_predict_training_set(df = test_df,
                                                                              input_array_col_names = self.analyze_descriptor_approach_inputs['molecular_descriptors'],
                                                                              output_array_col_names = self.analyze_descriptor_approach_inputs['output_label'],
                                                                              min_max_scalar = min_max_scalar,
                                                                              )
            model_storage[test_set] = {
                    'predict_df': predict_df,
                    'predict_stats': predict_stats,
                    }
        return model_storage
        
    ### FUNCTION TO GENERATE PARITY PLOTS
    def plot_parity( self, parity_plot_inputs, index=0):
        '''
        The purpose of this function is to plot parity plot. 
        INPUTS:
            parity_plot_inputs: [dict]
                dictionary of parity plot inputs, e.g.:
                    ## DEFINING INPUT TO PARITY PLOT
                    parity_plot_inputs = {
                            'save_fig_size' : (16.8/2, 16.8/2),
                            'fig_name' : figname + '.' + fig_extension,
                            'save_fig' : True,
                            }
            index: [int]
                index of parity plot that you want to plot
        OUTPUTS:
            fig, ax: figure and axis to parity plot
        '''
        ## ADDING TO DICTIONARY
        parity_plot_inputs['sigma_act_label'] = self.analyze_descriptor_approach_inputs['output_label']
        parity_plot_inputs['mass_frac_water_label'] = 'mass_frac_water'
        parity_plot_inputs['sigma_pred_label'] = 'y_pred'
        
        ## DEFINING MODEL 
        output_name = self.output_name_list[index]
        df = self.predict_df[index]
        
        ## PRINTING
        print("Printing model: %s"%( output_name ) )
        
        ## PLOTTING PARITY
        fig, ax = plot_parity_publication_single_solvent_system( dataframe = df,
                                                       **parity_plot_inputs)
        
        return fig, ax


#%%
## MAIN FUNCTION
if __name__ == "__main__":
    
    ## DEFINING CNN DICT
    cnn_dict = CNN_DICT
    ## DEFINING NUMBER OF EPOCHS
    cnn_dict['epochs'] = 500
    
    ## DEFINING FULL PATH TO MD DESCRIPTORS AND EXPERIMENTS
    path_md_descriptors=r"R:\scratch\3d_cnn_project\database\Experimental_Data\solvent_effects_regression_data_MD_Descriptor_with_Sigma.csv"
    ## SIMULATION PATH
    path_sim = r"R:\scratch\3d_cnn_project\simulations\FINAL\2B_md_descriptor_testing"

    #################
    ### MAIN CODE ### 
    #################
    
    ## CREATING LIST OF LAYERS
    neuron_list = [[10, 10, 10]]  # np.arange(1, 10)
            
    ## DEFINING INPUTS
    analyze_descriptor_approach_inputs={ 
             'path_md_descriptors': path_md_descriptors,
             'molecular_descriptors' : [ 'gamma', 'tau', 'delta' ],
             'output_label' : 'sigma_label',
             'verbose' : False,
            }
    
    ## DEFINING INPUTS
    nn_descriptors_model_inputs = {
            'path_md_descriptors': path_md_descriptors,
            'path_sim': path_sim,
            'neuron_list': neuron_list,
            'analyze_descriptor_approach_inputs': analyze_descriptor_approach_inputs,
            'nn_dict': cnn_dict,
            'learning_rate': 0.001,
            }
    
    ## DEFINING NN MODEL
    nn_model = nn_descriptors_model( **nn_descriptors_model_inputs )
    
    #%%
    nn_training_each_solvent = nn_model.train_across_data( column_name = 'cosolvent')
    nn_training_each_reactant = nn_model.train_across_data( column_name = 'solute')
    
    
    #%%
    ## CROSSVALIDATING
    # nn_model.cross_validate(column_name = 'cosolvent')
    nn_cross_valid_solutes = nn_model.cross_validate(column_name = 'solute')
    
    #%%
    ## FINDING RMSES FOR EACH 
    rmse_solutes = [ nn_cross_valid_solutes[each_key]['predict_stats'][0]['rmse'] for each_key in nn_cross_valid_solutes.keys()]
    
    
    #%%
    
    
    
    
    ## DEFINING EXTENSION
    figname = "Testing_descriptors"
    fig_extension = "png"
    
    ## DEFINING INPUT TO PARITY PLOT
    parity_plot_inputs = {
            'save_fig_size' : (16.8/2, 16.8/2),
            'fig_name' : figname + '.' + fig_extension,
            'save_fig' : True,
            }
    
    ## GENERAITNG PARITY PLOT
    nn_model.plot_parity(parity_plot_inputs = parity_plot_inputs,
                         )