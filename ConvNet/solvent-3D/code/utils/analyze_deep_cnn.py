# -*- coding: utf-8 -*-
"""
analyze_deep_cnn.py
The purpose of this script is to extract fully trained deep cnn models. This script 
will have the same structure as train_deep_cnn.py. 

Created on: 04/23/2019

Author(s):
    - Alex K. Chew (alexkchew@gmail.com)
FUNCTIONS:
    create_dataframe: 
        creates the panda databased used for plottings
    find_avg_std_predictions:
        finds average and standard deviations of the predictions
"""
## IMPORTING NECESSARY MODULES
import os
## IMPORTING PANDAS
import pandas as pd
## IMPORTING NUMPY
import numpy as np
## IMPORTING PICKLE
import pickle
##  TIME
import time
## IMPORTING SYS
import sys

## IMPORTING KERAS DETAILS
from keras.layers import Conv3D, MaxPool3D, Flatten, Dense
from keras.layers import Dropout, Input, BatchNormalization
from keras.losses import mean_squared_error
from keras.optimizers import Adam
from keras.models import Model
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint

from core.path import read_combined_name_directories, extract_combined_names_to_vars
## IMPORTING SKLEARN MODULES
#from sklearn.metrics import accuracy_score
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import explained_variance_score
#from sklearn.metrics import mean_absolute_error
#from sklearn.metrics import r2_score

## IMPORTING GLOBAL VARIABLES
from core.global_vars import SOLUTE_TO_TEMP_DICT, COSOLVENT_COLORS, DEFAULT_PATH_DICT, CNN_DICT
## CHECKING TOOLS
from core.check_tools import check_testing
## IMPORTING COMBINING ARRAYS
from combining_arrays import combine_instances
## IMPORTING PATH FUNCTIONS
from core.path import find_paths
## IMPORTING NOMENCLATURE
from core.nomenclature import extract_instance_names, read_combined_name, extract_representation_inputs, extract_sampling_inputs

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
from train_deep_cnn import train_deep_cnn, split_train_test_set

## IMPORTING MODULES
import matplotlib.pyplot as plt
from core.plotting_scripts import DEFAULT_FIG_INFO, LABELS_DICT, LINE_STYLE, update_ax_limits, AXIS_RANGES, SAVE_FIG_INFO, \
                                     change_axis_label_fonts, TICKS_DICT, get_cmap
                                     
## IMPORTING SKLEARN MODULES
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
# from statistics import mean
from matplotlib.offsetbox import AnchoredText

## IMPORTING METRICS TOOL
from core.ml_funcs import metrics


############################
##### ACCURACY METRICS #####
############################

### FUNCTION TO CREATE PANDAS DATAFRAME
def create_dataframe(instance_dict,
                     y_true,
                     y_pred,
                     y_pred_std):
    '''
    The purpose of this function is to create a dataframe as below:
        reactant / temperature /cosolvent / mass fraction / actual values / pred values / error
    INPUTS:
        instance_dict: [dict]
            list with dictionary of each instance
        y_true: [list]
            list of true vlaues
        y_pred: [list]
            list of predicted values
        y_pred_std: [list]
            list of standard deviations of the predictions
    OUTPUTS:
        dataframe: [pd.dataframe]
            pandas dataframe of your data
    '''
    ## CREATING A PANDAS DATAFRAME
    dataframe = pd.DataFrame(instance_dict)
    
    ## ADDING PREDICTED AND TRUE VALUES
    dataframe['y_true'] = pd.Series(y_true)
    dataframe['y_pred'] = pd.Series(y_pred)
    dataframe['y_pred_std'] = pd.Series(y_pred_std)
    
    ## SORTING COLUMN BY COSOLVENT, THEN SOLUTE
    dataframe = dataframe.sort_values(by=['cosolvent', 'solute'])
    return dataframe

### FUNCTION TO AVG PRED AND STD
def find_avg_std_predictions(instance_names, 
                             y_pred,
                             y_true,):
    '''
    The purpose of this function is to find the average and standard deviation 
    of a predicted model. This code finds the splitting instances, then tries 
    to find average based on the splitting. This code will also check to see if the 
    split actually makes sense -- done by seeing if the y_true split is correct. 
    INPUTS:
        instance_names: [list]
            list of instance names
        y_pred: [list]
            list of predicted values
        y_true: [list]
            list of true values
    OUTPUTS:
        y_pred_avg: [np.array, shape=(num_instances)]
            average predicted values 
        y_pred_std: [np.array, shape=(num_instances)]
            standard deviation of predicted values
        y_true_split: [np.array, shape=(num_instances)]
            y true values fater splitting
    '''
    ## GETTING NUMBER OF SPLITS
    total_instance_split = int( len(y_pred) / len(instance_names) )
    total_num_splits = len(y_pred) / total_instance_split
    ## DIVIDING Y PRED
    y_pred_split = np.split( y_pred, total_num_splits)
    y_true_split = np.split( y_true, total_num_splits)
    
    ## CHECKING IF SPLIT WAS RIGHT
    if np.all(y_true_split[0] == y_true_split[0][0] ) != True:
        print("There may be an error in splitting! Check the find_avg_std_predictions function!")
        print("Here's how the split for y_true is:")
        print(y_true_split)
        print("Pausing here so you can see this error!")
        print("Exiting!")
        time.sleep(5)
        sys.exit(1)
        
    ## FINDING PREDICTED AVERAGE
    y_pred_avg = np.mean(y_pred_split,axis=1)
    y_pred_std = np.std(y_pred_split,axis=1)
    
    ## FINDING UPDATED Y TRUE
    y_true_split = np.array(y_true_split)[:,1]
    return y_pred_avg, y_pred_std, y_true_split

### FUNCTION TO PLOT LEARNING CURVE
def plot_learning_curve( x, 
                         y,
                         x_label = 'Number of epochs',
                         y_label = 'Loss',
                         ax = None, 
                         fig = None, 
                         title = None,
                         loc = 'lower right',
                         save_fig = False,
                         fig_name="learning_curve",
                         fig_format="pdf",
                         ):
    '''
    The purpose of this function is to plot learning curves given x and y. 
    INPUTS:
        x: [array]
            x values, usually epochs
        y: [array]
            y values, typically loss, etc.
        x_label: [str, default = 'Number of epochs']
            x labels
        y_label: [str, default = 'Loss'
            y labels
        loc: [str, default: 'lower right']
            location of the legend
        fig_name: [str, default="learning_curve"]
            figure name you want to save into
        fig_format: [str, default="pdf"]
            figure format name
        title: [str, default=None]
            title for the figure
        fig, ax: figure and axis labels                
    OUTPUTS:
        fig, ax: figure and axis labels
    '''
    ## CHECKING IF FIGURE IS CREATED
    if ax == None or fig == None:
        fig = plt.figure(**DEFAULT_FIG_INFO) 
        ax = fig.add_subplot(111)
        ## DRAWING LABELS
        if title is not None:
            ax.set_title( title ,**LABELS_DICT)
        ax.set_xlabel(x_label,**LABELS_DICT)
        ax.set_ylabel(y_label,**LABELS_DICT)
        ## UPDATING IMAGE
        ax = update_ax_limits(ax, axis_ranges =  AXIS_RANGES['learning_curve'] )
    
        ## UPDATING AXIS TICKS
        ax = change_axis_label_fonts(ax, TICKS_DICT)
        
    ## PLOTTING
    ax.plot( x, y, '-', color='k', linewidth=2 ) # label = "Max tree depth: %s"%(max_depth)
    
    ## TIGHT LAYOUT
    fig.tight_layout()
    
    ## SEEING IF YOU WANT TO SAVE FIGURE
    if save_fig == True:
        fig.savefig(fig_name + '.' + fig_format , format=fig_format, **SAVE_FIG_INFO)

    return fig, ax

### FUNCTION TO PLOT PARITY PLOT
def plot_parity( true_values,
                 pred_values,
                 pred_std_values = None,
                 ax = None, 
                 fig = None, 
                 title = None,
                 loc = 'lower right',
                 save_fig = False,
                 fig_name="parity_plot",
                 fig_format="pdf",
                 ):
    ''' This function plots the parity plot between predicted and actual values '''
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
    ## PLOTTING ERROR B ARS
    if pred_std_values is not None:
        ax.errorbar( true_values, pred_values, yerr = pred_std_values, color = 'k', fmt = 'o', 
                    capsize=2, ) # linestyle="None" elinewidth=3, markeredgewidth=10 
    
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

    ## SETTING GRID
    ax.grid()
    ax.set_axisbelow(True)

    ## TIGHT LAYOUT
    fig.tight_layout()
    
    ## SEEING IF YOU WANT TO SAVE FIGURE
    if save_fig == True:
        fig.savefig(fig_name + '.' + fig_format , format=fig_format, **SAVE_FIG_INFO)

    return fig, ax

### FUNCTION TO PLOT PARITY PLOT
def plot_parity_cosolvents( true_values,
                            pred_values,
                            unique_cosolvent_names,
                            cosolvent_split_index,
                            pred_std_values = None,
                            ax = None, 
                            fig = None, 
                            title = None,
                            loc = 'upper left',
                            save_fig = False,
                            fig_name="parity_plot",
                            fig_format="pdf",
                            ):
    ''' 
    This function plots the parity plot between predicted and actual values.
    INPUTS:
        true_values: [np.array]
            true values as an array
        pred_values: [np.array]
            predicted values as an array
        unique_cosolvent_names: [np.array]
            unique cosolvent names, e.g. 'DIO', ...
        cosolvent_split_index: [list]
            list of indices for numpy array for each cosolvent
    OUTPUTS:
        fig, ax
    '''
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
    
    ## LOOPING THROUGH EACH SOLVENT NAME
    for idx, cosolvent_name in enumerate(unique_cosolvent_names):
        ## FINDING TRUE VALUES
        cosolvent_true_values = true_values[cosolvent_split_index[idx]]
        cosolvent_pred_values = pred_values[cosolvent_split_index[idx]]
        cosolvent_pred_values_std = pred_std_values[cosolvent_split_index[idx]]
        ## FINDING COLOR
        try:
            cosolvent_color = COSOLVENT_COLORS[cosolvent_name]
        except Exception:
            cosolvent_color = 'k'
        
        ax.scatter( cosolvent_true_values, cosolvent_pred_values, marker = 'o', color=cosolvent_color, linewidth=.1, label = cosolvent_name ) # label = "Max tree depth: %s"%(max_depth)
        ## PLOTTING ERROR B ARS
        if pred_std_values is not None:
            ax.errorbar( cosolvent_true_values, cosolvent_pred_values, yerr = cosolvent_pred_values_std, color=cosolvent_color, fmt = 'o', 
                        capsize=2, ) # linestyle="None" elinewidth=3, markeredgewidth=10 
    
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

    ## SETTING GRID
    ax.grid()
    ax.set_axisbelow(True)
    
    ## ADDING LEGEND
    ax.legend(loc = loc)
    
    ## TIGHT LAYOUT
    fig.tight_layout()
    
    ## SEEING IF YOU WANT TO SAVE FIGURE
    if save_fig == True:
        fig.savefig(fig_name + '.' + fig_format , format=fig_format, **SAVE_FIG_INFO)

    return fig, ax

###################################################
### CLASS FUNCTION TO ANALYZE DEEP CNN NETWORKS ###
###################################################
class analyze_deep_cnn:
    '''
    The purpose of this class is to analyze deep cnn networks. We will go through 
    multiple possible analysis tools using the data you supply. We have assumed 
    you trained the data set and are looking for how your 3D CNN worked out. 
    INPUTS:
        deep_cnn: [obj]
            fully trained deep cnn
    OUTPUTS:
        self.history: [dict]
            history dictionary of the learning details
        self.y_pred: [np.array]
            predictions of y values
        self.y_true: [np.array]
            true y-values
    FUNCTIONS:
        ## PLOTTING FUNCTIONS
        plot_learning_curve: plots learning curve
        plot_parity_scatter: plots parity scatter
        ## OTHER FUNCTIONS
        make_predictions_test_set: uses test set to make y_pred
        compute_regression_accuracy: [staticmethod] way to compute regression accuracy
    '''
    def __init__(self, instances, deep_cnn):
        
        ## STORING INSTANCE NAMES
        self.instance_names = instances.instance_names
        
        ## STORING TOTAL TIME IN SECONDS
        if deep_cnn.want_cross_validation is False:
            self.time_hms = deep_cnn.time_hms
        
        ## STORING OUTPUT PATH
        self.deep_cnn_output_path = deep_cnn.output_path
        
        ## STORING OUTPUT NAME
        self.deep_cnn_output_file_name = deep_cnn.output_file_name
        
        ## EXTRACTION FROM DEEP CNN
        if deep_cnn.want_cross_validation is False:
            self.history = deep_cnn.history_history
        
        ## GENERATING TEST SET
        self.x_test, self.y_true_raw = self.generate_test_set(instances = instances,
                                                              deep_cnn = deep_cnn)
        
        ## DICTIONARY
        self.model_y_pred_raw = []
        
        ## LOOPING THROUGH MODELS
        for idx, model in enumerate(deep_cnn.model_list):
        
            ## MAKING PREDICTIONS ON TEST SET
            self.y_pred_raw = self.make_predictions_test_set(x_test = self.x_test,
                                                              y_true = self.y_true_raw,
                                                              model = model)
        
            ## DEFINING ANALYSIS DIRECTIONARY
            self.model_y_pred_raw.append(self.y_pred_raw)
        
        ## AVERAGING ALL PREDICTIONS
        self.model_avg_y_pred = np.mean(self.model_y_pred_raw,axis=0)

        ## MAKING PREDICTIONS BY AVERAGING
        self.y_pred, self.y_pred_std , self.y_true = self.find_avg_std_predictions(instances = instances, 
                                                                                   deep_cnn = deep_cnn,
                                                                                   y_pred = self.model_avg_y_pred,
                                                                                   y_true = self.y_true_raw,)
        
        ## COMPUTING REGRESSION ACCURACY
        self.accuracy_dict = self.compute_regression_accuracy(y_pred = self.y_pred,
                                                              y_true = self.y_true,
                                                              )
        ## FINDING INSTANCE NAMES
        self.find_instance_names()
        
        ## FINDING COSOLVETN REGRESSION DATA
        self.compute_cosolvent_regression_data()
        
        ## CREATING DATAFRAME INFORMATION
        self.create_dataframe()
        
        return
    
    ### FUNCTION TO GENERATE TEST SET
    def generate_test_set(self, instances, deep_cnn):
        '''
        The purpose of this function is to generate the test set.
        INPUTS:
            deep_cnn: [obj]
                deep cnn object
            instances: [obj]
                instances object            
        OUTPUTS:
            
        '''
        
        if deep_cnn.want_cross_validation is False:
            x_test = deep_cnn.x_test
            y_true = deep_cnn.y_test
        else:
            ## GETTING ALL IMPORTANT DATA
            _, x_test, _, y_true = split_train_test_set(instances = instances, 
                                                        sampling_dict = deep_cnn.sampling_dict )
            
        ## TRYING
        try:
            ## SEEING IF DESCRIPTORS WAS USED
            if deep_cnn.want_descriptors is True:
                x_test = [ deep_cnn.x_test, deep_cnn.md_descriptor_list_test ]
        except Exception:
            pass
        return x_test, y_true
    
    ### FUNCTION TO TRY THE TEST SET AND MAKE PREDICTIONS
    def make_predictions_test_set(self, x_test, y_true, model):
        '''
        The purpose of this function is to make predictions of the test set 
        given the model. By default, we will make predictions on the test 
        set as it will give information about the model.
        INPUTS:
            x_test: [list]
                list of x testing data
            y_true: [list]
                list of the true values
            model: [obj, default=None]
                model to use. If None, it will look for the model from deep_cnn
        OUTPUTS:
            y_pred: [np.array]
                predictions of y values
        '''
        ## PREDICTIONS
        y_pred = model.predict(x_test).reshape(len(y_true) )
        
        return y_pred
    
    ### FUNCTION TO AVG PRED AND STD
    def find_avg_std_predictions(self, 
                                 instances, 
                                 deep_cnn,
                                 y_pred,
                                 y_true,):
        '''
        The purpose of this function is to find the average and standard deviation 
        of a predicted model. This code finds the splitting instances, then tries 
        to find average based on the splitting. This code will also check to see if the 
        split actually makes sense -- done by seeing if the y_true split is correct. 
        INPUTS:
            instances: [obj]
                object from combining arrays code
            deep_cnn: [obj]
                fully trained deep cnn
            y_pred: [np.array]
                y predicted array
            y_true: [np.array]
                y true array
        OUTPUTS:
            y_pred_avg: [np.array, shape=(num_instances)]
                average predicted values 
            y_pred_std: [np.array, shape=(num_instances)]
                standard deviation of predicted values
            y_true_split: [np.array, shape=(num_instances)]
                y true values fater splitting
        '''
        ## GETTING AVERAGE AND STANDARD DEVIATION OF PREDICTIONS
        y_pred_avg, y_pred_std, y_true_split = find_avg_std_predictions(instance_names =instances.instance_names,
                                                                         y_pred = y_pred,
                                                                         y_true = y_true)
        return y_pred_avg, y_pred_std, y_true_split

    ### FUNCTION TO COMPUTE REGRESSION ACCURACY
    @staticmethod
    def compute_regression_accuracy( y_pred, y_true ):
        '''
        The purpose of this code is to compute the regression accuracy. 
        INPUTS:
            y_pred: [np.array]
                predictions of y values
            y_true: [np.array]
                true y-values
        OUTPUTS:
            accuracy_dict: [dict]
                accuracy in the form of a dictionary
        '''
        ## COMPUTING ACCURACY (MEAN AVERAGE ERROR, ROOT MEAN SQUARED ERROR, etc.)
        accuracy_dict = metrics(y_fit = y_pred, 
                                y_act = y_true,
                                want_dict = True)
        
        return accuracy_dict
        
    ### FUNCTION TO PLOT LEARNING CURVE INFORMATION
    def plot_learning_curve(self, loss_type='loss', fig_name = "learning_curve", fig_format="pdf", save_fig=False):
        '''
        This function plots the learning curve for a specific type. 
        INPUTS:
            loss_type: [str]
                loss function type. You can check the different types in self.history.keys
            save_fig: [logical, default = False]
                True if you want to save the figure
            fig_name: [str, default="learning_curve"]
                figure name you want to save into
            fig_format: [str, default="pdf"]
                figure format name
        OUTPUTS:
            fig, ax: figure and axis for plot
        '''
        ## CHECKING IF THIS IS WITHIN HISTORY KEYS
        if loss_type in self.history.keys():
            ## GETTING NUMBER OF EPOCHS
            n_epochs = len(self.history[loss_type])
            x = np.arange(1, n_epochs+1)
            loss = self.history[loss_type]
            ## GENERATING PLOT
            fig, ax = plot_learning_curve( x = x, 
                                           y = loss,
                                           y_label = loss_type,
                                           fig_name = fig_name,
                                           save_fig = save_fig,
                                           fig_format = fig_format,
                                          )
        else:
            print("Error! Loss type is not available: %s"%( loss_type )  )
            fig, ax = None, None
        print("Available loss types: %s"%( ', '.join(self.history.keys())  ) )
        return fig, ax
    
    ### FUNCTION TO PLOT PARITY PLOT
    def plot_parity_scatter(self, fig_name="parity_plot", fig_format="pdf", save_fig=False, want_statistical_text_box = True):
        '''
        The purpose of this function is to plot the scatter parity between 
        predicted and actual values.
        INPUTS:
            want_statistical_text_box: [logical, default=True]
                True if you want statistics added to parity plot
        OUTPUTS:
            fig, ax: figure and axis for plot
        '''
        ## MAKING PLOT
        fig, ax = plot_parity( true_values = self.y_true,
                               pred_values = self.y_pred, 
                               pred_std_values = self.y_pred_std,
                               fig_name = fig_name,
                               save_fig = False,
                               fig_format = fig_format,
                              )
        if want_statistical_text_box == True:
            ## CREATING BOX TEXT
            box_text = "%s: %.2f\n%s: %.2f"%( "Slope", self.accuracy_dict['slope'],
                                              "RMSE", self.accuracy_dict['rmse']) 
            ## ADDING TEXT BOX
            text_box = AnchoredText(box_text, frameon=True, loc=4, pad=0.5)
            plt.setp(text_box.patch, facecolor='white', alpha=0.5)
            ax.add_artist(text_box)
            
        ## SEEING IF YOU WANT TO SAVE FIGURE
        if save_fig == True:
            fig.savefig(fig_name + '.' + fig_format , format=fig_format, **SAVE_FIG_INFO)
            
        return fig, ax
    
    
    ### FUNCTION TO FIND INSTANCE NAMES
    def find_instance_names(self, ):
        '''
        The purpose of this function is to extract the instance names. 
        INPUTS:
            void
        OUTPUTS:
            self.instance_dict: [dict]
                instance dictionary of extracted values
        '''
        ## LOOPING THROUGH INSTANCE NAMES AND GETTING DICTIONARY
        self.instance_dict = [ extract_instance_names(name = name) for name in self.instance_names ]
        
    ### FUNCTION TO FIND COSOLVENT REGRESSION DATA
    def compute_cosolvent_regression_data( self ):
        '''
        The purpose of this function is to find each cosolvent and get regression 
        data based on that. 
        INPUTS:
            void
        OUTPUTS:
            self.unique_cosolvent_names: [np.array]
                unique cosolvent names
            self.cosolvent_split_index: [list]
                list of cosolvent indices relative to instances
            self.cosolvent_true_values: [list]
                true value relative to each cosolvent
            self.cosolvent_pred_values: [list]
                true values relative to predicted values
            self.cosolvent_pred_values_std: [list]
                std true values
            self.cosolvent_regression_accuracy: [list]
                list of regression accuracy for each individual cosolvent
        '''
        ## FINDING ALL COSOLVENT NAMES
        instance_cosolvent_list = np.array([ each_dict['cosolvent'] for each_dict in self.instance_dict])
        
        ## FINDING ALL UNIQUE COSOLVENTS
        self.unique_cosolvent_names, cosolvent_index, cosolvent_count = np.unique(instance_cosolvent_list, return_inverse=True, return_counts = True)
        
        ## LOOPING THROUGH AND GETTING COSOLVENT SPLITS
        self.cosolvent_split_index = np.split(np.argsort(cosolvent_index), np.cumsum(cosolvent_count[:-1]))
        
        ## FINDING PREDICTED VALUES AND EXPERIMENTAL VALUES FOR EACH SPLIT
        self.cosolvent_true_values = []
        self.cosolvent_pred_values = []
        self.cosolvent_pred_values_std = []
        for idx, cosolvent_name in enumerate(self.unique_cosolvent_names):
            ## FINDING TRUE VALUES
            self.cosolvent_true_values.append(self.y_true[self.cosolvent_split_index[idx]])
            self.cosolvent_pred_values.append(self.y_pred[self.cosolvent_split_index[idx]])
            self.cosolvent_pred_values_std.append(self.y_pred_std[self.cosolvent_split_index[idx]])
        
        ## GENERATING DICTIONARY FOR REGRESSION ACCURACY PER COSOLVENT
        self.cosolvent_regression_accuracy = {}
        for idx, cosolvent_name in enumerate(self.unique_cosolvent_names):
            self.cosolvent_regression_accuracy[cosolvent_name] =self.compute_regression_accuracy( y_pred = self.cosolvent_pred_values[idx],
                                                                                                  y_true = self.cosolvent_true_values[idx] )
        return
        
    ### FUNCTION TO CREATE PANDAS DATAFRAME
    def create_dataframe(self):
        '''
        The purpose of this function is to create a dataframe as below:
            reactant / temperature /cosolvent / mass fraction / actual values / pred values / error
        INPUTS:
            void
        OUTPUTS:
            self.dataframe: [pd.dataframe]
                pandas dataframe of your data
        '''
        ## CREATING DATAFRAME
        self.dataframe = create_dataframe(instance_dict = self.instance_dict,
                                          y_true = self.y_true,
                                          y_pred = self.y_pred,
                                          y_pred_std = self.y_pred_std)
        return
    
    ### FUNCTION TO PLOT PARITY COSOLVENT
    def plot_parity_scatter_cosolvent(self, 
                                      fig_name="parity_plot_cosolvent", 
                                      fig_format="pdf", 
                                      save_fig=False, 
                                      want_statistical_text_box = True):
        '''
        The purpose of this function is to plot the scatter parity between 
        predicted and actual values.
        INPUTS:
            want_statistical_text_box: [logical, default=True]
                True if you want statistics added to parity plot
        OUTPUTS:
            fig, ax: figure and axis for plot
        '''
        ## PLOTTING
        fig, ax = plot_parity_cosolvents( 
                                    true_values = self.y_true,
                                    pred_values = self.y_pred,
                                    pred_std_values = self.y_pred_std,
                                    fig_name = fig_name,
                                    save_fig = False,
                                    fig_format = fig_format,
                                    unique_cosolvent_names = self.unique_cosolvent_names,
                                    cosolvent_split_index = self.cosolvent_split_index,
                )
        
        if want_statistical_text_box == True:
            ## CREATING BOX TEXT
            box_text = "%s: %.2f\n%s: %.2f"%( "Slope", self.accuracy_dict['slope'],
                                              "RMSE", self.accuracy_dict['rmse']) 
            ## ADDING TEXT BOX
            text_box = AnchoredText(box_text, frameon=True, loc=4, pad=0.5)
            plt.setp(text_box.patch, facecolor='white', alpha=0.5)
            ax.add_artist(text_box)
            
        ## SEEING IF YOU WANT TO SAVE FIGURE
        if save_fig == True:
            fig.savefig(fig_name + '.' + fig_format , format=fig_format, **SAVE_FIG_INFO)
            
        return fig, ax
    
    ### FUNCTION TO STORE PICKLE
    def store_pickle(self, results_file_path = None):
        ''' This function stores the pickle'''
        ## DEFINING LOCATION TO STORE THE PICKLE
        if results_file_path == None:
            try:
                ## FINDING PATHS
                path_dict = find_paths()
                ## DEFINING RESULTS PATH
                results_file_path = path_dict['result_path']
            except Exception:
                results_file_path = self.deep_cnn_output_path 
        self.pickle_location = os.path.join( results_file_path , self.deep_cnn_output_file_name + ".results" )
        print("Creating results pickle file in: %s"%( self.pickle_location ) )
        with open(self.pickle_location, 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump([self], f, protocol=2)  # <-- protocol 2 required for python2   # -1
    
    

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
        solvent_list = [ 'DIO', 'GVL', 'THF' ]# 'GVL', 'THF' ] , 'GVL', 'THF'
        ## DEFINING MASS FRACTION DATA
        mass_frac_data = ['10', '25', '50', '75']
        ## DEFINING SOLUTE LIST
        solute_list = list(SOLUTE_TO_TEMP_DICT)
        ## DEFINING TYPE OF REPRESENTATION
        representation_type = 'split_avg_nonorm' # split_avg_nonorm
        representation_inputs = {
                'num_splits': 5
                }
        ## DEFINING PATHS
        database_path = None # Since None, we will find them!
        class_file_path = None
        combined_database_path = None
        output_file_path = None # r"C:\Users\akchew\Box Sync\2019_Spring\CS760\Spring_2019_CS760_Project\Output\30_30_30" # None # OUTPUT PATH FOR CNN NETWORKS
        
        ## DEFINING SIMULATION DIRECTORY
        simulation_path = r"R:\scratch\3d_cnn_project\simulations\190625-20_20_20_VoxNet_Descriptor_Testing"
        
        ## DEFINING SIMULATION
        simulation_dir = r"20_20_20_32ns_first-split_avg_nonorm-8-strlearn-0.75-voxnet-500-CEL_ETBE_FRU_LGA_PDO_XYL_tBuOH-10_25_50_75-DIO_GVL_THF-MD"
        
        ## DEFINING VERBOSITY
        verbose = True
        
        ## DEFINING RETRAINING DETAILS
        retrain = False
        
        ## DEFINING SAMPLING INFORMATION
        sampling_dict = {
                'name': 'stratified_learning',
                'split_training': 3,
                }
        
        ## SELECTING TYPE OF CNN
        cnn_type = 'solvent_net' # 'orion' # 'solvent_net' 'voxnet'
        
        ## DEFINING NUMBER OF EPOCHS
        cnn_dict = {
                'epochs': 500,
                'validation_split': 0.2,
                'batch_size': 18, # higher batches results in faster training, lower batches can converge faster
                'metrics': ['mean_squared_error']
                }
        
        ## DEFINING DATA TYPE
        data_type="20_20_20" # 30_30_x30
        
       
    ## EXTRACTING DETAILS
    current_directory_extracted = read_combined_name( simulation_dir )
    ## EXTRACTING INFORMATION
    representation_type, \
    representation_inputs, \
    sampling_dict, \
    data_type, \
    cnn_type, \
    num_epochs, \
    solute_list, \
    solvent_list, \
    mass_frac_data, want_descriptor = extract_combined_names_to_vars(extracted_name = current_directory_extracted)
        
        
    #%%
    
    #-------------- MAIN FUNCTION
    ## LOADING THE DATA
    instances = combine_instances(
                     solute_list = solute_list,
                     representation_type = representation_type,
                     representation_inputs = representation_inputs,
                     solvent_list = solvent_list, 
                     mass_frac_data = mass_frac_data, 
                     verbose = verbose,
                     database_path = DEFAULT_PATH_DICT['database_path'],
                     class_file_path = DEFAULT_PATH_DICT['class_file_path'],
                     combined_database_path = DEFAULT_PATH_DICT['combined_database_path'],
                     data_type = data_type
                     )
    
    #%%
    
    ### TRAINING CNN
    deep_cnn = train_deep_cnn(
                     instances = instances,
                     sampling_dict = sampling_dict,
                     cnn_type = cnn_type,
                     cnn_dict = cnn_dict,
                     retrain=retrain,
                     output_path = os.path.join(simulation_path,simulation_dir),
                     verbose = verbose,
                     want_basic_name = True,
                     want_descriptors = want_descriptor,
                     )



    #%%

    ## WANT TO SAVE FIGURE?
    save_fig = False # False    
    
    ## ANALYZING
    analysis = analyze_deep_cnn( instances = instances,
                                 deep_cnn = deep_cnn )
    
    #%%
    ## STORING ANALYSIS
    analysis.store_pickle( results_file_path = os.path.join(simulation_path,simulation_dir) )
    
    #%%
    ## PRINTING RMSE
#    print("Slope: %.3f"%(analysis.accuracy_dict['slope']))
#    print("RMSE: %.3f"%(analysis.accuracy_dict['rmse']))
#
#    ## PLOTTING LEARNING CURVE
#    fig, ax = analysis.plot_learning_curve(loss_type="loss", 
#                                           fig_name = deep_cnn.output_file_name + "-learning_curve" ,
#                                           save_fig=save_fig, 
#                                           fig_format = "png") # val_mean_squared_error
#    ## PLOTTING PARITY SCATTER
#    fig, ax = analysis.plot_parity_scatter(save_fig = save_fig, 
#                                           fig_name = deep_cnn.output_file_name + "-parity_plot" ,
#                                           fig_format = "png")
#    
#    ## PLOTTING PARITY SCATTER
#    fig, ax = analysis.plot_parity_scatter_cosolvent(save_fig = True, 
#                                           fig_name = deep_cnn.output_file_name + "-parity_cosolvent_plot" ,
#                                           fig_format = "png")
    
    ## DATA FRAME OBJECT: analysis.dataframe
    
    