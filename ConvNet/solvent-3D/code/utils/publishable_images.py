# -*- coding: utf-8 -*-
"""
publishable_images.py
The purpose of this code is to generate publishable images. 

Created on: 06/17/2019

Author(s):
    - Alex K. Chew (alexkchew@gmail.com)

"""

## IMPORTING PYTHON
import pandas as pd
## NUMPY
import numpy as np
## OS
import os
## PICKLE
import pickle

import matplotlib as mpl
mpl.rcParams['axes.linewidth'] = 1.0
mpl.rcParams['xtick.major.width'] = 1.0
mpl.rcParams['ytick.major.width'] = 1.0

## DEFINING FIGURE SIZES IN CM (SUPPORTING INFORMATION)
FIGURE_SIZES_CM_SI={
        '1_col': np.array([ 8.3, 8.3 ]),
        '2_col': np.array([ 17.1, 17.1 ]),
        }

## DEFINING LINESTYLE
LINESTYLE={
        'linewidth': 1.5,
        }

## IMPORTING MODULES
import core.plotting_scripts as plotter
## DEFINING GLOBAL VARIABLES
from core.global_vars import CNN_DICT
## IMPORTING ANALYSIS TOOL
from analyze_deep_cnn import analyze_deep_cnn, metrics, create_dataframe, find_avg_std_predictions

## GETTING PEARSON'S R
from scipy.stats import pearsonr

## CROSS VALIDATION MODULE
from read_cross_validation import analyze_cross_validation, plot_all_cross_validations, compute_cumulative_rmse, compute_stats_from_cross_valid, get_test_set_df_from_cross_valid

## IMPORTING MODULES
from deep_ann_descriptors import nn_descriptors_model

## PICKLE FUNCTIONS
from extraction_scripts import load_pickle_general

## PREDICTIVE MODEL
from prediction_post_training import predict_with_trained_model, get_test_pred_test_database_dict
# TEST_DATABASE_DICT
## PLOTTING PARITY PLOT
from read_extract_deep_cnn import plot_parity_publication_single_solvent_system, generate_dataframe_slope_rmse

## DEFINING NOMENCLATURE
from core.nomenclature import read_combined_name, extract_representation_inputs
from combining_arrays import combine_instances


## MODULE TO VARY TIME SIZES
from read_sampling_time_increment_varying_training_size import read_sampling_time_increments_with_varying_training_sizes

## SAMPLING TIME CHUNKS
from read_sampling_time_chunks import read_sampling_time_chunks

###############################################################################
### GLOBAL VARIABLES
###############################################################################

## DEFINING MAIN DIRECTORY PATH
MAIN_DIR_PATH=r"R:\scratch\3d_cnn_project"
# r"R:/scratch/storage/2019_3d_cnn/3d_cnn_project"

## DEFINING IMPORTANT PATHS
path_image_dir= r"C:\Users\akchew\Box Sync\VanLehnGroup\2.Research Documents\Alex_RVL_Meetings\20200210\images\3d_cnn"
#  os.path.join(MAIN_DIR_PATH, "images")
# r"C:\Users\akchew\Box Sync\VanLehnGroup\0.Manuscripts\Solvent_effects_3D_CNNs\Images"

## DEFINING FIG EXTENSION
fig_extension = 'png'
# 'svg' # 'svg'

## DEFINING LOGICALS
save_fig = True
# True 
# # True

## DEFINING DEFAULT PATHS
database_path = os.path.join(MAIN_DIR_PATH, "database")
## DEFINING EXPERIMENTAL DATA PATH
exp_data_path = os.path.join(database_path, "Experimental_Data")

sim_image_dir = os.path.join(MAIN_DIR_PATH, "images")
class_file_path = os.path.join(exp_data_path, "solvent_effects_regression_data.csv")
combined_database_path =  os.path.join(MAIN_DIR_PATH, r"combined_data_set")
## DEFINING LOCATION TO STORE PICKLE
path_pickle = os.path.join(MAIN_DIR_PATH, "storage")
## DEFINING SIM PATH
sim_path = os.path.join(MAIN_DIR_PATH, "simulations")
## DEFINING PATH TO OUTPUT EXCEL SPREADSHEET
path_output_excel_spreadsheet = os.path.join(MAIN_DIR_PATH, "csv_output")
# r"C:\Users\akchew\Box Sync\VanLehnGroup\0.Manuscripts\Solvent_effects_3D_CNNs\Excel_Spreadsheet\csv_output"

## DEFINING PATH TO NEURAL NETWORKS WITH DESCRIPORS
MD_DESCRIPTOR_PATH = os.path.join( sim_path, "FINAL")

## DEFINING PATH DICTIONARY
path_dict = {
        'database_path': database_path,
        'class_file_path': class_file_path,
        'combined_database_path': combined_database_path,
        'path_image_dir': path_image_dir,
        'sim_path': sim_path,
        'path_pickle': path_pickle,
        'sim_image_dir': path_image_dir,
        'path_md_descriptors': os.path.join(exp_data_path, "solvent_effects_regression_data_MD_Descriptor_with_Sigma.csv"),
        'path_md_descriptors_regression': os.path.join(exp_data_path, "solvent_effects_MD_prediction_model_all_data_regression.csv"),
        'path_comparison': os.path.join(exp_data_path, "solvent_effects_comparison_between_models.csv"),
        'path_md_descriptors_nn': os.path.join(sim_path, r"2B_md_descriptor_nn_final")
        }

## DEFINING SIMULATION PATHS
simulation_path_dicts={
        '3D_CNN_Training_All_Solvents': os.path.join( path_dict['sim_path'], '2020203-5fold_train_20_20_20_20ns_oxy_3chan' ), # '20200131-5fold_train_20_20_20_20ns_oxy_3chan'
        'Increment_varying_training': os.path.join(  path_dict['sim_path'], r"190804-training_size_3chan" ), # r"190706-solvent_net_sample_increment_varying_training_size"
        'Sampling_chunks_training': os.path.join( path_dict['sim_path'], r"190804-solvent_net_sampling_chunks_20ns"), #  r"190708-solvent_net_sampling_chunks_20ns" 
        'VGG16': os.path.join( path_dict['sim_path'], '190802-parity_vgg16_32_32_32_20ns_oxy_3chan_firstwith10' ),
        }

## ADDING NEW PATHS FOR SOLVENT NET, ORION, AND VOXNET
simulation_path_dicts['3D_CNN_Training_All_Solvents_Solvent_Net'] = os.path.join(simulation_path_dicts['3D_CNN_Training_All_Solvents'],
                                                                                 '20_20_20_20ns_oxy_3chan-split_avg_nonorm-10-strlearn-0.80-solvent_net-500-CEL_ETBE_FRU_LGA_PDO_XYL_tBuOH-10_25_50_75-DIO_GVL_THF')

simulation_path_dicts['3D_CNN_Training_All_Solvents_orion'] = os.path.join(simulation_path_dicts['3D_CNN_Training_All_Solvents'],
                                                                                 '20_20_20_20ns_oxy_3chan-split_avg_nonorm-10-strlearn-0.80-orion-500-CEL_ETBE_FRU_LGA_PDO_XYL_tBuOH-10_25_50_75-DIO_GVL_THF')

simulation_path_dicts['3D_CNN_Training_All_Solvents_voxnet'] = os.path.join(simulation_path_dicts['3D_CNN_Training_All_Solvents'],
                                                                                 '20_20_20_20ns_oxy_3chan-split_avg_nonorm-10-strlearn-0.80-voxnet-500-CEL_ETBE_FRU_LGA_PDO_XYL_tBuOH-10_25_50_75-DIO_GVL_THF')

# VGG16
simulation_path_dicts['3D_CNN_Training_All_Solvents_vgg16'] = os.path.join(simulation_path_dicts['VGG16'],
                                                                                 r'32_32_32_20ns_oxy_3chan_firstwith10-split_avg_nonorm_planar-10-strlearn-0.80-vgg16-500-CEL_ETBE_FRU_LGA_PDO_XYL_tBuOH-10_25_50_75-DIO_GVL_THF')

## CROSS VALIDATION PATHS 
cross_validation_paths = {
        'VoxNet_Solute': r'20200203-cross_val_size-20_20_20_20ns_oxy_3chan-voxnet-solute', 
        'VoxNet_Cosolvent': r'20200203-cross_val_size-20_20_20_20ns_oxy_3chan-voxnet-cosolvent',
        'ORION_Solute': r'20200203-cross_val_size-20_20_20_20ns_oxy_3chan-orion-solute', 
        'ORION_Cosolvent': r'20200203-cross_val_size-20_20_20_20ns_oxy_3chan-orion-cosolvent',
        'SolventNet_Solute': r'20200203-cross_val_size-20_20_20_20ns_oxy_3chan-solvent_net-solute',
        'SolventNet_Cosolvent': r'20200203-cross_val_size-20_20_20_20ns_oxy_3chan-solvent_net-cosolvent',
        'NN_Solute': path_dict['path_md_descriptors_nn'],
        'NN_Cosolvent': path_dict['path_md_descriptors_nn'],
        }

## SI CROSS VALIDATION PATHS
SI_cross_validation_path = {
        'vgg16_Solute': r"190802-cross_val_planar-32_32_32_20ns_oxy_3chan_firstwith10-vgg16-solute",
        'vgg16_Cosolvent': r"190802-cross_val_planar-32_32_32_20ns_oxy_3chan_firstwith10-vgg16-cosolvent"
        }

## ADDING TO EACH PATH
cross_validation_paths = { each_key: os.path.join( path_dict['sim_path'], cross_validation_paths[each_key] ) for each_key in cross_validation_paths.keys()}
# cross_validation_different_reps_paths = { each_key: os.path.join( path_dict['sim_path'], cross_validation_different_reps_paths[each_key] ) for each_key in cross_validation_different_reps_paths.keys()}
SI_cross_validation_paths = { each_key: os.path.join( path_dict['sim_path'], SI_cross_validation_path[each_key] ) for each_key in SI_cross_validation_path.keys()}

## ADDING TO SIMULATION PATHS
simulation_path_dicts['cross_validation_paths'] = cross_validation_paths
simulation_path_dicts['SI_cross_validation_paths'] = SI_cross_validation_path
# simulation_path_dicts['cross_validation_different_reps_paths'] = cross_validation_different_reps_paths



''' OLD PATHS

## DEFINING PATH TO NEURAL NETWORKS WITH DESCRIPORS
MD_DESCRIPTOR_PATH = os.path.join( sim_path, "FINAL")

## DEFINING PATH DICTIONARY
path_dict = {
        'database_path': database_path,
        'class_file_path': class_file_path,
        'combined_database_path': combined_database_path,
        'path_image_dir': path_image_dir,
        'sim_path': sim_path,
        'path_pickle': path_pickle,
        'sim_image_dir': sim_image_dir,
        'path_md_descriptors': os.path.join(exp_data_path, "solvent_effects_regression_data_MD_Descriptor_with_Sigma.csv"),
        'path_md_descriptors_regression': os.path.join(exp_data_path, "solvent_effects_MD_prediction_model_all_data_regression.csv"),
        'path_comparison': os.path.join(exp_data_path, "solvent_effects_comparison_between_models.csv"),
        'path_md_descriptors_nn': os.path.join(sim_path, r"2B_md_descriptor_nn_final")
        }

## DEFINING SIMULATION PATHS
simulation_path_dicts={
        '3D_CNN_Training_All_Solvents': os.path.join( path_dict['sim_path'], '190725-newrep_20_20_20_20ns_oxy_3chan' ),
        '3D_CNN_Training_VoxNet_ORION_All_Solvents': os.path.join( path_dict['sim_path'], '190802-parity_diff_systems_20_20_20_20ns_oxy_3chan' ),
        'Increment_varying_training': os.path.join(  path_dict['sim_path'], r"190804-training_size_3chan" ), # r"190706-solvent_net_sample_increment_varying_training_size"
        
        'Sampling_chunks_training': os.path.join( path_dict['sim_path'], r"190804-solvent_net_sampling_chunks_20ns"), #  r"190708-solvent_net_sampling_chunks_20ns" 
        'VGG16': os.path.join( path_dict['sim_path'], '190802-parity_vgg16_32_32_32_20ns_oxy_3chan_firstwith10' ),
        }

## ADDING NEW PATHS FOR SOLVENT NET, ORION, AND VOXNET
simulation_path_dicts['3D_CNN_Training_All_Solvents_Solvent_Net'] = os.path.join(simulation_path_dicts['3D_CNN_Training_All_Solvents'],
                                                                                 '20_20_20_20ns_oxy_3chan-split_avg_nonorm-10-strlearn-0.80-solvent_net-500-CEL_ETBE_FRU_LGA_PDO_XYL_tBuOH-10_25_50_75-DIO_GVL_THF')

simulation_path_dicts['3D_CNN_Training_All_Solvents_orion'] = os.path.join(simulation_path_dicts['3D_CNN_Training_VoxNet_ORION_All_Solvents'],
                                                                                 '20_20_20_20ns_oxy_3chan-split_avg_nonorm-10-strlearn-0.80-orion-500-CEL_ETBE_FRU_LGA_PDO_XYL_tBuOH-10_25_50_75-DIO_GVL_THF')

simulation_path_dicts['3D_CNN_Training_All_Solvents_voxnet'] = os.path.join(simulation_path_dicts['3D_CNN_Training_VoxNet_ORION_All_Solvents'],
                                                                                 '20_20_20_20ns_oxy_3chan-split_avg_nonorm-10-strlearn-0.80-voxnet-500-CEL_ETBE_FRU_LGA_PDO_XYL_tBuOH-10_25_50_75-DIO_GVL_THF')

# VGG16
simulation_path_dicts['3D_CNN_Training_All_Solvents_vgg16'] = os.path.join(simulation_path_dicts['VGG16'],
                                                                                 r'32_32_32_20ns_oxy_3chan_firstwith10-split_avg_nonorm_planar-10-strlearn-0.80-vgg16-500-CEL_ETBE_FRU_LGA_PDO_XYL_tBuOH-10_25_50_75-DIO_GVL_THF')

## CROSS VALIDATION PATHS 
cross_validation_paths = {
        'VoxNet_Solute': r'190802-cross_val-20_20_20_20ns_oxy_3chan-voxnet-solute', 
        'VoxNet_Cosolvent': r'190802-cross_val-20_20_20_20ns_oxy_3chan-voxnet-cosolvent',
        'ORION_Solute': r'190802-cross_val-20_20_20_20ns_oxy_3chan-orion-solute', 
        'ORION_Cosolvent': r'190802-cross_val-20_20_20_20ns_oxy_3chan-orion-cosolvent',
        'SolventNet_Solute': r'190725-cross_valid-newrep-20_20_20_20ns_oxy_3chan-solvent_net-solute',
        'SolventNet_Cosolvent': r'190725-cross_valid-newrep-20_20_20_20ns_oxy_3chan-solvent_net-cosolvent',
        'NN_Solute': path_dict['path_md_descriptors_nn'],
        'NN_Cosolvent': path_dict['path_md_descriptors_nn'],
        }

## SI CROSS VALIDATION PATHS
SI_cross_validation_path = {
        'vgg16_Solute': r"190802-cross_val_planar-32_32_32_20ns_oxy_3chan_firstwith10-vgg16-solute",
        'vgg16_Cosolvent': r"190802-cross_val_planar-32_32_32_20ns_oxy_3chan_firstwith10-vgg16-cosolvent"
        }

## ADDING TO EACH PATH
cross_validation_paths = { each_key: os.path.join( path_dict['sim_path'], cross_validation_paths[each_key] ) for each_key in cross_validation_paths.keys()}
# cross_validation_different_reps_paths = { each_key: os.path.join( path_dict['sim_path'], cross_validation_different_reps_paths[each_key] ) for each_key in cross_validation_different_reps_paths.keys()}
SI_cross_validation_paths = { each_key: os.path.join( path_dict['sim_path'], SI_cross_validation_path[each_key] ) for each_key in SI_cross_validation_path.keys()}

## ADDING TO SIMULATION PATHS
simulation_path_dicts['cross_validation_paths'] = cross_validation_paths
simulation_path_dicts['SI_cross_validation_paths'] = SI_cross_validation_path
# simulation_path_dicts['cross_validation_different_reps_paths'] = cross_validation_different_reps_paths



'''
###########################################################
### DEFINING DEFAULT DICTIONARY FOR DESCRIPTOR INPUTS
###########################################################
## CREATING LIST OF LAYERS
NEURON_LIST = [[10, 10, 10]]

## DEFINING CNN DICT
cnn_dict = CNN_DICT
## DEFINING NUMBER OF EPOCHS
cnn_dict['epochs'] = 500

ANALYZE_DESCRIPTOR_APPROACH_INPUTS={ 
         'path_md_descriptors': path_dict['path_md_descriptors'],
         'molecular_descriptors' : [ 'gamma', 'tau', 'delta' ],
         'output_label' : 'sigma_label',
         'verbose' : False,
        }

NN_DESCRIPTOR_MODEL_INPUTS = {
        'path_md_descriptors': path_dict['path_md_descriptors'],
        'path_sim': path_dict['path_md_descriptors_nn'],
        'neuron_list': NEURON_LIST,
        'analyze_descriptor_approach_inputs': ANALYZE_DESCRIPTOR_APPROACH_INPUTS,
        'nn_dict': cnn_dict,
        'learning_rate': 0.001,
        }

###############################################################################
### FUNCTIONS AND CLASSES 
###############################################################################

## FUNCTION TO RENAME COSOLVENTS
def rename_dataframe_entries(dataframe):
    ''' This function renames entries of a dataframe'''
    ## RENAMING TBA ENTRIES
    if dataframe['solute'].str.match('tBuOH').any() == True:
        dataframe = plotter.rename_df_column_entries(df = dataframe,
                                      col_name = 'solute',
                                      change_col_list = [ 'tBuOH', 'TBA'  ],
                                      )
    ## REORDERING DF
    dataframe = plotter.order_df(df = dataframe,
                 ordered_classes = plotter.SOLUTE_ORDER,
                 col_name = 'solute',
                  )
        
    return dataframe

## FUNCTION THAT CORRECTS MASS FRACTION OF WATER
def correct_mass_frac_water( mass_frac_label ):
    '''
    The purpose of this function is to correct mass fraction label
    '''
    if mass_frac_label > 1:
        mass_frac_water = "%.2f"%( float(mass_frac_label)/100.)
    else:
        mass_frac_water = "%.2f"%(mass_frac_label)
    return mass_frac_water

### FUNCTION THAT DEALS WITH SAVING FIGURE
def store_figure(fig, path, fig_extension = 'png', save_fig=False, dpi=1200):
    '''
    The purpose of this function is to store a figure.
    INPUTS:
        fig: [object]
            figure object
        path: [str]
            path to location you want to save the figure (without extension)
        fig_extension:
    OUTPUTS:
        void
    '''
    ## STORING FIGURE
    if save_fig is True:
        ## DEFINING FIGURE NAME
        fig_name =  path + '.' + fig_extension
        print("Printing figure: %s"%(fig_name) )
        fig.savefig( fig_name, 
                     format=fig_extension, 
                     dpi = dpi,    
                     )
    return

## FINDING METRIC OF EACH PER COLUMN
def compute_df_metric_per_column( df,
                                column_name = 'solute',
                                y_true_label = 'y_true',
                                y_pred_label = 'y_pred',
                                current_metric = "rmse",
                                ):
    '''
    The purpose of this function is to compute rmse of each data set based 
    on column name.
    INPUTS:
        df: [pd.dataframe]
            pandas dataframe containing all the information
        column_name: [str]
            name of the column
        y_true_label: [str]
            true label for y values
        y_pred_label: [str]
            predicted labels for y values
        current_metric: [str]
            metric that you want
                "rmse": root mean squared error
                "slope": slope of the data
                "pearson": pearsons r of the data
    OUTPUTS:
        output_dict: [dict]
            dictionary containing metric of each column entry
    
    '''    
    ## FINDING ALL UNIQUE SOLUTES
    unique_names = np.unique(df[column_name])
    
    ## DEFINING ALL RMSE
    output_dict = {}
    
    ## LOOPING THROUGH EACH UNIQUE NAME
    for each_name in unique_names:
        ## LOCATING ALL COLUMNS
        data_frame_of_column = df[df[column_name] == each_name]
        ## EXTRACTING ALL Y PRED AND Y TRUE
        y_pred = np.array(data_frame_of_column[y_pred_label])
        y_true = np.array(data_frame_of_column[y_true_label])
        ## DEFINING METRIC
        if current_metric == "rmse":
            ## COMPUTING RMSE
            output_value = metrics( y_fit = y_pred,y_act = y_true )[1]
        elif current_metric == "slope":
            output_value = metrics( y_fit = y_pred,y_act = y_true )[4]
        elif current_metric == "pearson" or current_metric == 'pearson_r':
            output_value = pearsonr( x = y_true, y = y_pred )[0]
        else:
            print("Error! Current metric (%s) is not defined! Check compute_df_metric_per_column"%(current_metric))
        ## STORING RMSE
        output_dict[each_name] = output_value
    return output_dict

## FUNCTION TO FIX FIGURE AXIS AND LIMITS
def fix_figure_ticks_and_lims(ax,
                              x_ticks = None,
                              y_ticks = None,
                              x_lims = None,
                              y_lims = None,
                              ):
    '''
    The purpose of this function is to fix the figure ticks and ax limits 
    INPUTS:
        ax: [object]
            figure axis
        x_ticks: [tuple, size 3]
            tuple containing minumum, maximum, and increments of the x ticks, e.g. (0, 1.00, 0.2 )
        y_ticks: [tuple, size 3]
            tuple containing minumum, maximum, and increments of the y ticks, e.g. (-1.5, 2.5, 0.5 )
        x_lims: [tuple, size 2]
            tuple containing minimum and maximum of x limits, e.g. (-0.1, 1.1)
        y_lims: [tuple, size 2]
            tuple containing minimum and maximum of y limits, e.g. (-0.5, 2.5)
    '''
    ## SETTING X TICKS AND Y TICKS
    if x_ticks is not None:
        ax.set_xticks(np.arange(x_ticks[0], x_ticks[1] + x_ticks[2], x_ticks[2]))
    if y_ticks is not None:
        ax.set_yticks(np.arange(y_ticks[0], y_ticks[1] + y_ticks[2], y_ticks[2]))
    ## SETTING X Y LIMS
    if x_lims is not None:
        ax.set_xlim([x_lims[0], x_lims[1]] )
    if y_lims is not None:
        ax.set_ylim([y_lims[0], y_lims[1]])
    return

### FUNCTION TO MAKE A PLOT FOR SAMPLING TIME INCREMETNS
def publish_plot_sampling_time_increments( read_sampling_times, 
                                           figure_details,
                                           amount_ns_per_partition = 10,
                                           ):
    '''
    The purpose of this function is to plot RMSE vs. amount of training time
    INPUTS:
        read_sampling_times: [object]
            class object that contains all sampling time information
        figure_details: [dict]
            dictionary containing all figure details
        amount_ns_per_partition: [float, default=10 ns]
            amount of nanoseconds per partition
    OUTPUTS:
        fig, ax: figure and axis for the plot
    '''
    ## DEFINING FIGURE
    fig, ax = plotter.create_fig_based_on_cm( figure_details['figsize'] )
    
    ## GETTING CMAP COLORS
    cmap = plotter.get_cmap(  len(read_sampling_times.sim_rmse_storage) )
    
    ## FIXING PLOT FIGURE AND TICKS
    fix_figure_ticks_and_lims(ax = ax,
                              **figure_details['figure_limits']
                              )
    ## ADDING Y LINE AT 0.10
    ax.axhline( y = 0.10, linestyle = '--', color = 'k', linewidth=2)
    
    ## SETTING AXIS LABELS
    ax.set_xlabel("Simulation time per partition (ns)")
    ax.set_ylabel("RMSE")
    
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
        ax.plot(x, y, '.-', color = current_color, label = label, **LINESTYLE )
    
    ## ADDING LEGEND
    ax.legend()
    
    ## SHOW TIGHT LAYOUT
    fig.tight_layout()
    
    return fig, ax

### FUNCTION TO PLOT CUMULATIVE SUM
def plot_cumulative_rmse_cross_valid( cross_valid_dict,
                                      figure_details,
                                      n_bins = 50,
                                      ):
    '''
    The purpose of this function is to plot cumulative RMSE across validations
    INPUTS:
        cross_valid_dict: [dict]
            dictionary of cross valdiation results
        figure_details: [dict]
            dictionary for figure details
        n_bins: [int]
            number of bins
    OUTPUTS:
        fig, ax: 
            figure and axis for the plot
    '''
    ## GETTING LIST
    cross_valid_list = list(cross_valid_dict)
    
    ## DEFINING FIGURE
    fig, ax = plotter.create_fig_based_on_cm( figure_details['figsize'] )
    ## GETTING CMAP COLORS
    cmap = plotter.get_cmap(  len(cross_valid_list) )
    
    ## FIXING PLOT FIGURE AND TICKS
    fix_figure_ticks_and_lims(ax = ax,
                              **figure_details['figure_limits']
                              )
    
    ## SETTING AXIS LABELS
    ax.set_xlabel("Root-mean-squared error")
    ax.set_ylabel("Cumulative probability distribution")
    
    ## LOOPING THROUGH EACH
    for idx, key in enumerate(cross_valid_list):
        ## COMPUTING CUMULATIVE RMSE
        cumulative_RMSE = compute_cumulative_rmse( cross_valid_dict[key] )
        
        ## DEFINING COLOR
        current_color = cmap(idx)
        
        ## PLOTTING CUMULATIVE HISTOGRAM
        n, bins, patches = ax.hist(cumulative_RMSE, n_bins, density=True, histtype='step', color = current_color,
                                   cumulative=True, label = key, **LINESTYLE)
    
    ## ADDING LEGEND
    ax.legend(loc="lower center")
    
    ## SHOW TIGHT LAYOUT
    fig.tight_layout()
    
    return fig, ax

########################################################
### CROSS VALIDATION
########################################################    

## CLASS FUNCTION TO GENERATE CROSS VALIDATION RESULTS
class extract_cross_validation:
    '''
    The purpose of this function is to set up cross validation so that we can 
    publish the results.
    INPUTS:
        null
    OUTPUTS:
        extracted ctross validations
    FUNCTIONS:
        load_cross_validation: loads cross validation
        plot_parity_plot: plots parity plot
        get_df_multiple_cross_validations: finds dataframe cross validation data
        
    '''
    ## INITIALIZING
    def __init__(self):
        
        ## CREATING EMPTY DATAFRAME
        self.df_storage = []
        
    ### FUNCTION TO LOAD CROSS VALIDATION RESULTS
    @staticmethod
    def load_cross_validation(cross_valid_inputs, 
                              pickle_path = path_dict['path_pickle'],
                              NN_dict = None,
                              current_key = None,
                              path_to_sim = None):
        '''
        The purpose of this function is to load the cross validation results
        INPUTS:
            cross_valid_inputs: [dict]
                dictionary with cross validation inputs -- should include a 
                'main_dir' key with the main directory that your cross validation is in.
            pickle_path: [str]
                path to pickle file for storage
            current_key: [str, default = None]
                key to cross validation extraction, e.g.
                    'NN_Solute' or 'NN_Cosolvent' will change how the loading is done
            path_to_sim: [str]
                path to sim, used for NN_solute
        OUTPUTS:
            cross_valid_results: cross validation results
        '''
        ## DEFINING MAIN DIRECTORY
        main_dir = cross_valid_inputs['main_dir']
        
        ## DEFINING PICKLE STORAGE NAME
        pickle_storage_name =  os.path.join( path_dict['path_pickle'], main_dir + "_storage.pickle")
        
        ## REDEFINING PICKLE STORAGE LOCATION
        if NN_dict is not None:
            ## GETTING COLUMN NAME
            column_name = NN_dict['column_name']
            pickle_storage_name =  os.path.join( path_dict['path_pickle'], main_dir + "_" + column_name + "_storage.pickle")
        
        ## RUNNING CROSS VALIDATION RESULTS
        if os.path.isfile(pickle_storage_name) is not True:
            ## SEEING IF YOU WANT NEURAL NETWORK
            if NN_dict is not None:
                nn_descriptor_model_inputs = {**NN_dict}
                nn_descriptor_model_inputs.pop("column_name", None) 
            
                ## DEFINING NN MODEL
                nn_model = nn_descriptors_model( **nn_descriptor_model_inputs )
                
                ## CROSS VALIDATING
                cross_valid_results = nn_model.cross_validate(column_name = column_name)
            else:
                ## NORMAL CROSS VALIDATION
                cross_valid_results = analyze_cross_validation( **cross_valid_inputs )
            ## STORING PICKLE
            with open(pickle_storage_name, 'wb') as f:  # Python 3: open(..., 'wb')
                pickle.dump([cross_valid_results], f, protocol=2)  # <-- protocol 2 required for python2   # -1
        ## RESTORING
        else:
            cross_valid_results = load_pickle_general(pickle_storage_name)[0]
        return cross_valid_results
    
    ### FUNCTION TO PLOT PARITY PLOT
    @staticmethod
    def plot_parity_plot(cross_valid_results, parity_plot_inputs, want_combined_plot = True):
        '''
        The purpose of this function is to plot the parity plot
        INPUTS:
            parity_plot_inputs: [dict]
                dictionary partiy plot inputs
            want_combined_plot: [dict]
                True if you want a combined plot (default)
        '''    
        ## ADDING THE PARITY PLOT DICTIONARY
        parity_plot_inputs['sigma_act_label'] = 'y_true'
        parity_plot_inputs['sigma_pred_label'] = 'y_pred'
        parity_plot_inputs['sigma_pred_err_label'] = 'y_pred_std'
        parity_plot_inputs['mass_frac_water_label'] = 'mass_frac'
        
        ## PLOTTING ALL CROSS VALIDATIONS
        fig, ax = plot_all_cross_validations( cross_validation_storage = cross_valid_results.cross_validation_storage, 
                                    parity_plot_inputs = parity_plot_inputs,
                                    want_combined_plot = True)
        return fig, ax
        
    ### FUNCTION TO LOAD MULTIPLE CROSS VALIDATIONS
    def load_multiple_cross_validations(self,
                                        path_to_cross_validation,
                                        pickle_path = path_dict['path_pickle'],
                                        cross_valid_inputs = {},
                                        ):
        '''
        The purpose of this function is to load multiple cross validations.
        INPUTS:
            path_to_cross_validation: [dict]
                dictionary path to cross validation
            pickle_path: [str]
                path to store the pickle
            cross_valid_inputs: [dict]
                cross validation inputs
        OUTPUTS:
            cross_validation_storage: [dict]
                dictionary storing cross validation results
        '''
        ## CREATING EMPTY DATAFRAME
        cross_validation_storage = {}
        
        ## LOOPING THROUGH THE KEYS
        for each_key in path_to_cross_validation:
            ## DEFINING MAIN DIRECTORY
            main_dir = os.path.basename(path_to_cross_validation[each_key])
            print("Working on cross validation for: %s"%(each_key) )
            
            ## CHANGING MAIN DIRECTORY IN CROS VALIDATION
            cross_valid_inputs['main_dir'] = main_dir
            
            ## DEFINING PICKLE STORAGE NAME
            pickle_storage_name =  os.path.join(pickle_path , main_dir + "_storage.pickle")
            
            ### SEEING IF NEURAL NETWORK IS PART OF THE KEYS
            if each_key == 'NN_Solute' or each_key == 'NN_Cosolvent':
                
                ## FIGURE OUT WHICH CROSS YOU WANT
                if 'Solute' in each_key:
                    column_name = 'solute'
                elif 'Cosolvent' in each_key:
                    column_name = 'cosolvent'
                else:
                    print("Error! Column name %s not defined!"%( each_key ) )
                
                ## PRINTING 
                print("Working on column name: %s"%(column_name) )
                NN_dict = {**NN_DESCRIPTOR_MODEL_INPUTS}
                ## ADDING TO COLUMN
                NN_dict['column_name'] = column_name
                ## ADDING SIM PATH
                NN_dict['path_sim'] = path_to_cross_validation[each_key]
                
                ## REDEFINING PICKLE STORAGE LOCATION
                pickle_storage_name =  os.path.join( path_dict['path_pickle'], main_dir + "_" + column_name + "_storage.pickle")
            else:
                NN_dict = None
            
            ## PRINTING PICKLE SOTRAGE PATH
            print("----------------------------------------------")
            print("Pickle storage path: %s"%(pickle_storage_name) )
            
            ## DEFINING CROSS VALIDATION RESULTS
            cross_valid_results = self.load_cross_validation(cross_valid_inputs = cross_valid_inputs,
                                                        pickle_path = pickle_path,
                                                        NN_dict = NN_dict
                                                        )
            ## APPENDING
            cross_validation_storage[each_key] = cross_valid_results
            
        return cross_validation_storage
            
    
    ### FUNCTION TO GET DFS FOR MULTIPLE CROSS VALIDATIONS
    def get_df_multiple_cross_validations(self,
                                          cross_validation_storage,
                                          output_csv_name = "output",
                                          pickle_path = path_dict['path_pickle'],
                                          path_output_excel_spreadsheet = path_output_excel_spreadsheet,
                                          desired_stats = ['slope', 'rmse', 'pearson_r'],
                                          ):
        '''
        The purpose of this function is to extract cross validation for a 
        given a dictionary of paths
        INPUTS:
            path_to_cross_validation: [dict]
                dictionary path to cross validation
            cross_valid_inputs: [dict]
                cross validation inputs used to extract details
            cross_valid_inputs: [str]
                output csv name
            pickle_path: [str]
                path to store the pickle
            NN_dict: [dict]
                dictionary for neural network inputs
            path_output_excel_spreadsheet: [str]
                path to output spreadsheet
            desired_stats: [list]
                list of desired stats
        OUTPUTS:
            df_storage: [list]
                list containing dataframe information
        '''
        df_storage = []
        ## LOOPING THROUGH EACH KEY
        for each_key in cross_validation_storage:
            ## DEFINING CROSS VALIDATION
            cross_valid_results = cross_validation_storage[each_key]
            
            ## EXTRACTING EACH CROSS VALIDATION
            ## DEFINING EACH CROSS VALIDATION
            if each_key == 'NN_Solute' or each_key == 'NN_Cosolvent':
                df = extract_cross_valid_nn( cross_valid_storage = cross_valid_results, 
                                             desired_stats = desired_stats,
                                             desired_name = each_key )
            else:
                df = extract_cross_valid_rmse(cross_valid_storage = cross_valid_results.cross_validation_storage,
                                              desired_stats = desired_stats,
                                              desired_name = each_key,
                                              )
            ## STORING DF
            df_storage.append(df)
        
        ## DEFINING OUTPUT CSV FILE
        path_output_cross_csv = os.path.join(path_output_excel_spreadsheet, output_csv_name)
        
        ## PRINTING
        self.print_csv_solute_and_cosolvent(cross_validation_storage = cross_validation_storage,
                                            df_storage = df_storage,
                                            path_output_cross_csv = path_output_cross_csv,
                                            type_list = ['Solute', 'Cosolvent'],
                                            want_concat = True)
        
        return df_storage
    
    ### FUNCTION TO GET STATISTICS BASED ON SPECIFIC stats
    def get_df_test_set_stats(self,
                              cross_validation_storage,
                              desired_stats = ['slope', 'rmse', 'pearson_r'],
                              path_output_excel_spreadsheet = path_output_excel_spreadsheet,
                              output_csv_name = "output",
                              ):
        '''
        The purpose of this function is to get all dataframe statistics 
        based on your input cross validation results. 
        INPUTS:
            cross_validation_storage: [dict]
                dictionary containing all cross validation storage information
            desired_stats: [list]
                list of stats names that you want
            path_output_excel_spreadsheet: [str]
                path to output spreadsheet
            output_csv_name: [str]
                output csv name
        OUTPUTS:
            df_storage: [list]
                list containing dataframe information
        '''
        ## CREATING EMPTY DATAFRAME
        df_storage = []
        
        ## LOOPING THROUGH EACH KEY
        for each_key in cross_validation_storage:
            ## DEFINING CROSS VALIDATION REULTS
            cross_valid_results = cross_validation_storage[each_key]
            ## SETTING NN TO NONE
            NN_inputs = None
            ## DEFINING PREDICTION KEYS
            y_pred_key = 'y_pred'
            y_true_key = 'y_true'
            ## SEEING IF NN IN KEY
            if each_key == 'NN_Solute' or each_key == 'NN_Cosolvent':
                ## GETTING DICT
                NN_inputs = {}
                ## DEFINING TRUE VALUES
                y_pred_key = 'y_pred'
                y_true_key = 'sigma_label'
                ## DEFINING INPUTS
                if each_key == 'NN_Solute':
                    NN_inputs['cross_validation_name'] = 'solute'
                elif each_key == 'NN_Cosolvent':
                    NN_inputs['cross_validation_name'] = 'cosolvent'           
                
            
            ## GETTING TEST DATABASE
            test_set_df_full = get_test_set_df_from_cross_valid(cross_valid_results,
                                                                NN_inputs = NN_inputs)
            ## COMPUTING STATISTICS
            output_stats = compute_stats_from_cross_valid(test_set_df_full = test_set_df_full, 
                                                          desired_stats = desired_stats,
                                                          y_pred_key = y_pred_key,
                                                          y_true_key = y_true_key
                                                          )
            ## APPENDING
            df_storage.append(output_stats)
            
        ## DEFINING OUTPUT CSV FILE
        path_output_cross_csv = os.path.join(path_output_excel_spreadsheet, output_csv_name)
        
        ## PRINTING
        self.print_csv_solute_and_cosolvent(cross_validation_storage = cross_validation_storage,
                                            df_storage = df_storage,
                                            path_output_cross_csv = path_output_cross_csv,
                                            type_list = ['Solute', 'Cosolvent'])
        
        return df_storage
    
    ### FUNCTION TO LOOP THROUGH EACH AND PRINT
    @staticmethod
    def print_csv_solute_and_cosolvent(cross_validation_storage,
                                       df_storage,
                                       path_output_cross_csv,
                                       type_list = ['Solute', 'Cosolvent'],
                                       want_concat = False):
        '''
        The purpose of this function is to print the csv of a solute and cosolvent 
        cross validation dataframe storage.
        INPUTS:
            cross_validation_storage: [dict]
                cross validation storage details
            df_storage: [list]
                list of dataframe results
            path_output_cross_csv: [str]
                path to output csv
            type_list: [list]
                type list of interest
            want_concat: [logical, default = False]
                True if you want to concatenate your storage (without keys added on)
        OUTPUTS:
            null, prints out output file
        '''
        ## STORING FOR SOLUTES AND COSOLVENTS
        for each_type in type_list:
            ## FINDING ALL KEYS
            idx_storage = [ idx for idx, each_key in enumerate(cross_validation_storage.keys()) if each_type in each_key]
            ## DEFINING DICT
            if want_concat is False:
                dict_storage = { list(cross_validation_storage.keys())[idx]: df_storage[idx] for idx in idx_storage}
                ## GETTING PANDAS
                df_dict_storage = pd.DataFrame(dict_storage)
            else:
                ## CONCENTATING
                dict_storage = [  df_storage[idx] for idx in idx_storage ]
                df_dict_storage = pd.concat(dict_storage, sort=False)
            ## ADDING TO CSV
            each_type_csv_path = path_output_cross_csv + '_' + each_type + '.csv'
            ## PRINTING
            print("Storing cross valid data in: %s"%(each_type_csv_path) )
            ## STORING
            df_dict_storage.to_csv(each_type_csv_path)
        return
        

    
########################################################
### LEARNING CURVES
########################################################
    
## DEFINING COLORS
LEARNING_CURVE_NETWORK_COLORS= \
    { 'solvent_net':
            {
                    'color': 'k',
                    },
       'orion':
           {
                   'color': 'b'
                   },
        'voxnet':
            {
                    'color': 'r'
                    },
        'vgg16':
            {
                    'color': 'k'
                    }
     }
            
## DEFINING LOSS TYPES
LEARNING_CURVE_LOSS_TYPES = \
    {   'loss':
            {
                    'linestyle': '-',
                    },
        'val_loss':
            {
                    'linestyle': '--',
                    },
            }

## CLASS FUNCTION TO PLOT LEARNING CURVE
class publish_plot_learning_curve:
    '''
    The purpose of this function is to plot learning curve. 
    INPUTS:

        figure_size: [tuple]
            figure size in centimeters
    OUTPUTS:
        
    '''

    ## INITIALIZING
    def __init__(self, 
                 figure_size = ( 8.3, 8.3 ),
                 ):
        ## CREATING FIGURE
        self.fig, self.ax = plotter.create_fig_based_on_cm( figure_size )
        
        ## SETTING AXIS LABELS
        self.ax.set_xlabel("Number of epochs")
        self.ax.set_ylabel("Loss")
        
        return
    
    ### FUNCTION TO PLOT EACH LOSS TYPE PER NETWORK COLOR
    def plot_each_loss_per_color(self, 
                                 history, 
                                 loss_types = LEARNING_CURVE_LOSS_TYPES,
                                 network_type = None,
                                 network_colors = LEARNING_CURVE_NETWORK_COLORS,
                                 linestyle = LINESTYLE,
                                 color = 'k'):
        '''
        Function to plot loss on a color basis
        INPUTS:
            history: [dict]
                history dictionary of your model
            loss_types: [dict]
                dictionary containing different losses linked to history. The dictionary 
                should contain specifications, like line styles, etc.
            network_name: [str]
                network name. If None, we will simply use black
            network_colors: [dict]
                dictionary containing each network color
            linestyle: [dict]
                dictionary containing the line style
        '''
        ## DEFINING NUMBER OF EPOCHS
        n_epochs = len(history[list(loss_types.keys())[0]])
        x = np.arange(1, n_epochs+1)
        
        ## DEFINING THE COLOR
        if network_type is not None:
            color = network_colors[network_type]['color']
            
        ## LOOPING THROUGH EACH LOSS
        for each_loss in loss_types:
            ## DEFINING EACH LOSS
            loss = history[each_loss]
            ## DEFINING PLOT STYLES
            plot_styles = loss_types[each_loss]
            ## ADDING CUSTOM FOR VGG16
            if network_type == 'vgg16':
                plot_styles = {'linestyle': '-',}
                if each_loss == 'loss':
                    color ='k'
                elif each_loss == 'val_loss':
                    color ='r'
            ## DEFINING PLOT STYLE LIST
            plot_style_dict = {**plot_styles, **LINESTYLE}
            plot_style_dict['color'] = color
            plot_style_dict['label'] = each_loss
            ## PLOTTING
            self.ax.plot( x, loss, **plot_style_dict )

        return
    ### FUNCTION TO ADD LEGEND
    def finalize_image(self):
        '''
        This is post correction for the image, e.g. adding legends and so on.
        INPUTS:
            self: [obj]
                class object
        OUTPUTS:
            void
        '''
        ## PLOTTING LEGEND
        self.ax.legend()
        self.fig.tight_layout()
        return

########################################################
### PREDICTIVE MODEL
########################################################
        
### FUNCTION TO EXTRACT PREDICTIVE MODEL
class publish_predict_model:
    '''
    The purpose of this class is to generate a predictive model.
    INPUTS:
        void
    OUTPUTS:
        void
    FUNCTIONS:
        loop_database_and_predict: loops through database and makes predictions
        generate_csv_stats: generates csv statistics
        plot_parity_plot: function to plot parity plot
    '''
    ## INITIALIZING
    def __init__(self):
        
        ## DEFINING EMPTY LIST FOR INITIALIZAITON
        self.csv_stats_storage = []
        
        return
    
    
    ### FUNCTION TO LOOP THROUGH DATABASE SETS TO MAKE PREDICTION
    def loop_database_and_predict(self,
                                  trained_model,
                                  main_dir,
                                  database_dict,
                                  path_pickle = path_dict['path_pickle'],
                                  want_repickle = False,
                                  num_partitions = 2):
        '''
        The purpose of this function is to loop through the database and 
        make predictions
        INPUTS:
            main_dir: [str]
                main directory name (used to store details)
            trained_model: [obj]
                training model
            path_pickle: [str]
                path of the pickle
            database_dict: [dict]
                dictionary that contains the path to the databases
            want_repickle: [logical, default=False]
                True if you want to repickle no matter what was done previously
            num_partitions: [int]
                number of partitions to use for prediction (earliest is used)
                desired metrics for the model
            fig_name_prefix: [str]
                string with figure name prefix to store
        OUTPUTS:
            stored_predicted_value_list_storage: [list]
                list of predicted values stored
            figure_name_list: [list]
                figure name list
        '''
        ## DEFINING STORAGE PREDICTIONS
        stored_predicted_value_list_storage = []
        
        ## STORING FIGURE NAMES
        figure_name_list = []
        
        ## LOOPING THROUGH THE DATABASE LIST
        for desired_database in database_dict:        
            
            ## DEFINING DATABASE
            path_test_database = database_dict[desired_database]['path_database']
            ## DEFINING PATH TO EXPERIMENTAL DATA
            path_exp_data = database_dict[desired_database]['path_exp_data']         
            ## DEFINING FIGURE NAME
            fig_name = "%s_%s"%(main_dir, desired_database)
            ## DEFINING STORING PREDICTED NAME
            predicted_value_name_storage = os.path.join( path_pickle, fig_name + "_storage.pickle")
            ## PRINTING
            print("*** Checking path to storage: %s"%(predicted_value_name_storage))
            ## PREDICTING THE VALUES (slow step)
            if os.path.isfile(predicted_value_name_storage) is not True or want_repickle is True:
                print("Path not found! Rerunning prediction model")
                ## PREDICTING THE TEST SETS
                stored_predicted_value_list = trained_model.predict_test_set(
                                                                                path_test_database = path_test_database,
                                                                                num_partitions = num_partitions,
                                                                                )
                ## ADDING TEST SET EXPERIMENTAL VALUES
                trained_model.add_test_set_exp_values( stored_predicted_value_list = stored_predicted_value_list,
                                                       path_exp_data = path_exp_data)
                ## STORING PICKLE
                with open(predicted_value_name_storage, 'wb') as f:  # Python 3: open(..., 'wb')
                    pickle.dump([stored_predicted_value_list], f, protocol=2)  # <-- protocol 2 required for python2   # -1
            ## RESTORING
            else:
                stored_predicted_value_list = load_pickle_general(predicted_value_name_storage)[0]
            
            ## APPENDING 
            stored_predicted_value_list_storage.append(stored_predicted_value_list.copy())
            figure_name_list.append(fig_name)
        return stored_predicted_value_list_storage, figure_name_list

    ### FUNCTION TO LOOP THROUGH MULTIPLE DATABASES
    def loop_multiple_simulation_paths(self,
                                       simulation_path_dict,
                                       loop_database_and_pred_inputs = {
                                            'path_pickle': path_pickle,
                                            'want_repickle': False,
                                            'num_partitions': 2,
                                               },
                                      model_weights = "model.hdf5",
                                       ):
        '''
        The purpose of this function is to loop through multiple simulation paths 
        and make predictions per database basis. 
        INPUTS:
            simulation_path_dict: [dict]
                dictionary containing all simulation paths
            loop_database_and_pred_inputs: [dict]
                dictionary for inputs for looping database
            model_weights: [str]
                name of the model weights
        OUTPUTS:
            predict_storage_dict: [dict]
                dictionary with predicted values per each key in simulation_path_dict
        '''
        ## CREATING DICTIONARY
        predict_storage_dict = {}
        
        ## LOOPING THROUGH EACH DIRECTORY
        for each_parity in simulation_path_dict:
            
            ## ADDING LABEL
            predict_storage_dict[each_parity] = {}
            
            ## DEFINING PARITY PLOT
            parity_path = simulation_path_dict[each_parity]
            
            ## DEFINING MAIN DIRECTORY
            main_dir = os.path.basename(parity_path)
            
            ## DEFINING FULL PATH TO MODEL
            path_model = os.path.join(parity_path, model_weights)
        
            ## DEFINING INPUTS FOR PREDICTED MODEL
            inputs_predicted_model = {
                    'path_model': path_model,
                    'verbose': True,
                    }
            ## LOADING MODEL
            trained_model = predict_with_trained_model( **inputs_predicted_model )
    
            ## DEFINING DATABASE
            test_database_basename = main_dir.split('-')[0] + '_'
            
            ## GETTING DATABASE DICT
            database_dict = get_test_pred_test_database_dict(test_database_basename = test_database_basename)
    
            ## DEFINING INPUTS FOR LOOP GRID
            loop_database_and_pred_inputs['trained_model'] = trained_model
            loop_database_and_pred_inputs['database_dict'] = database_dict
            loop_database_and_pred_inputs['main_dir'] = each_parity
            
            ## LOOPING DATABASE AND PREDICTING
            stored_predicted_value_list_storage, figure_name_list = self.loop_database_and_predict(**loop_database_and_pred_inputs)
            ## STORING
            ## LOOPING AND GETTING A DATAFRAME
            df_stored_predicted_value = [ pd.DataFrame(each_dict_list) for each_dict_list in stored_predicted_value_list_storage ]
            
            ## DEFINING STORAGE DICTIONARY
            predict_storage_dict[each_parity] = {
                    'df_stored_predicted_value' : df_stored_predicted_value,
                    'figure_name_list': figure_name_list,
                    'database_dict_keys': database_dict,
                    }
        
        return predict_storage_dict
    
    
    ### FUNCTION TO GENERATE CSV STATS FOR MULTIPLE PREDICTIONS
    def generate_csv_stats_multiple(self,
                                    predict_storage_dict,
                                    csv_file_name,
                                    path_output_excel_spreadsheet = path_output_excel_spreadsheet,
                                    desired_metrics = [  "slope","rmse", "pearson_r" ]):
        '''
        The purpose of this script is to generate multiple csv stats
        INPUTS:
            predict_storage_dict: [dict]
                dictionary of storage details from loop_multiple_simulation_paths function
            csv_file_name: [str]
                name of the output csv file
            path_output_excel_spreadsheet: [str]
                path to output excel sheet
            desired_metrics: [list]
                list of desired metrics
        OUTPUTS:
            void, outputs csv file
        '''
        ## CREATING DATAFRAMES
        data_storage = { each_metric: pd.DataFrame() for each_metric in desired_metrics } 
        ## LOOPING THROUGH EACH DATABASE
        for each_key in predict_storage_dict:
            ## CREATING STATS DICT
            overall_stats_dict = {each_key: {}}
            ## DEFINING PREDICTED VALUES
            stored_predicted_df = predict_storage_dict[each_key]['df_stored_predicted_value']
            ## COMPUTING OVERALL RMSE
            stored_predicted_df_overall = pd.concat(stored_predicted_df)
            overall_output_stats = compute_stats_from_cross_valid( test_set_df_full = stored_predicted_df_overall,
                                                                   desired_stats = desired_metrics,
                                                                   y_pred_key = 'y_pred',
                                                                   y_true_key = 'y_true',)
            
            ## LOOPING EACH METRIC
            for metric_index, current_metrics in enumerate(desired_metrics):
                            
                ## DEFINING STORAGE
                storage_metrics_dict = {}
                ## LOOPING THROUGH EACH DATABASE
                for idx, each_database_key in enumerate(predict_storage_dict[each_key]['database_dict_keys']):
                    ## DEFINING DATAFRAME
                    df_each_key = stored_predicted_df[idx]
    
                    ## COMPUTING STATISTICS
                    stats_dict = compute_df_metric_per_column( df = df_each_key,
                                                               column_name = 'solute',
                                                               y_true_label = 'y_true',
                                                               y_pred_label = 'y_pred',
                                                               current_metric = current_metrics,
                                                               )
                    ## AVERAGING DICT
                    avg_value = np.average([stats_dict[stats_key] for stats_key in stats_dict])
                    ## STORING
                    storage_metrics_dict[each_database_key] = avg_value
                ## DEFINING STORAGE DICT
                overall_stats_dict[each_key][current_metrics] = storage_metrics_dict
            
            ## STORING EACH STATS
            for idx_metric, each_metric in enumerate(desired_metrics):
                ## DEFINING STATS
                stats_dict = overall_stats_dict[each_key][each_metric]
                ## CREATING DATAFRAME
                df_stats_dict = pd.DataFrame( {each_key: stats_dict} ).T
                ## ADDING OVERALL
                df_stats_dict['overall'] = overall_output_stats[each_metric]
                ## APPENDING
                data_storage[each_metric] = data_storage[each_metric].append(df_stats_dict)
            
        ## POST DATA STORAGE, PRINTING CSV
        for idx_metric, each_metric in enumerate(desired_metrics):
            ## DEFINING CSV PATH
            path_csv_output = os.path.join( path_output_excel_spreadsheet,
                                            csv_file_name + '_' + each_metric + '.csv'
                                            )
            ## PRINTING
            print("Creating database for: %s"%(path_csv_output))
            data_storage[each_metric].to_csv(path_csv_output)
        
        return
    
    
    ### FUNCTION TO GENERATE CSV FILE
    def generate_csv_stats(self,
                           stored_predicted_value_list_storage,
                           database_dict,
                           figure_name_list,
                           path_output_excel_spreadsheet = None,
                           desired_metrics = [  "slope","rmse", "pearson" ],
                           want_csv = True,
                           ):
        '''
        The purpose of this function is to generate csv and the statistics from 
        the predictive model. This assumes you already made the predictions.
        INPUTS:
            self: [obj]
                class object
            path_output_excel_spreadsheet: [str]
                string path to output excel spreadsheet
            desired_metrics: [list]
                metrics you want to store
        OUTPUTS:
            df_stat_storage: [list of pd.database]
                database for statistics
        '''
        ## STORING AND DIRECTORIES
        database_list = database_dict.keys()
        ## DEFINING STORAGE
        df_stat_storage = []
        ## LOOPING THROUGH THE DATABASE LIST
        for idx, desired_database in enumerate(database_list):
            ## DEFINING PREDICTED VALUES
            stored_predicted_value_list = stored_predicted_value_list_storage[idx]
            ## DEFINING FIGURE NAME
            fig_name = figure_name_list[idx]
            ## CREATING DATAFRAME
            df = pd.DataFrame( stored_predicted_value_list )
            ## DEFINING STATISTICS DICTIONARY
            stats_dict = {current_metrics:  compute_df_metric_per_column( df = df,
                                                      column_name = 'solute',
                                                      y_true_label = 'y_true',
                                                      y_pred_label = 'y_pred',
                                                      current_metric = current_metrics,
                                                      )
                            for current_metrics in desired_metrics }
            ## CREATING A DATAFRAME
            df_stats = pd.DataFrame( stats_dict )

            ## APPENDING
            df_stat_storage.append(df_stats)

            ## IF WE WANT CSV IS TRUE
            if want_csv is True:
                ## DEFINING OUTPUT CSV FILE
                path_output_csv = os.path.join(path_output_excel_spreadsheet, fig_name + ".csv")
                df_stats.to_csv(path_output_csv)
                print("Writing statistics to %s"%(path_output_csv) )
        return df_stat_storage
    
    ### FUNCTION TO LOOP THROUGH CSV STATS
    def generate_csv_stats_given_list( self,
                                       train_model,
                                       loop_database_and_pred_inputs,
                                       csv_stats_inputs):
        '''
        The purpose of this function is to loop through each csv stats and outputs 
        the prediction
        INPUTS:
            self: [obj]
                class object
            train_model: [obj]
                trained model object from predict_with_train_model
            loop_database_and_pred_inputs: [dict]
                dictionary for inputs of loop database
            csv_stats_inputs: [dict]
                csv stats inputs
        OUTPUTS:
            self.csv_stats_storage: [list]
                csv stats storage
                
        '''
        ## GETTING BASNAME OF SIM
        sim_basename = os.path.basename(os.path.dirname(train_model.path_model))
        print("Working on: %s"%(sim_basename) )
        
        ## DEFINING DATABASE
        test_database_basename = sim_basename.split('-')[0] + '_' #  '20_20_20_20ns_oxy_3chan_'
        
        ## GETTING DATABASE DICT
        database_dict = get_test_pred_test_database_dict(test_database_basename = test_database_basename)
        
        ## STORING INPUTS
        loop_database_and_pred_inputs['database_dict'] = database_dict
        loop_database_and_pred_inputs['trained_model'] = trained_model
        
        ## LOOPING GRID
        stored_predicted_value_list_storage, figure_name_list = self.loop_database_and_predict(**loop_database_and_pred_inputs)
                                                                    
        ## CREATING INPUTS FOR CSV STATS
        csv_stats_inputs['stored_predicted_value_list_storage'] = stored_predicted_value_list_storage
        csv_stats_inputs['database_dict'] = database_dict
        csv_stats_inputs['figure_name_list'] = figure_name_list
        
        ## GETTING STATS
        df_stats = self.generate_csv_stats(**csv_stats_inputs)
        
        ## STORING IN DATA STATS
        self.csv_stats_storage.append(df_stats)

        return
    
    ### FUNCTION TO PLOT PARITY PLOT
    def plot_parity_plot(self, 
                         stored_predicted_value_list_storage,
                         parity_plot_inputs,
                         figure_name_list,
                         output_path = path_dict['sim_image_dir'],
                         save_fig = True,
                         fig_name_prefix = '5_predicted_model_',
                         ):
        '''
        The purpose of this function is to plot the parity plot for predicted 
        values. We assume you already made predictions from the loop_database_and_predict method. 
        INPUTS:
            self: [obj]
                class object
            parity_plot_inputs: [dict]
                inputs for parity plots, e.g.
                    {
                            'sigma_act_label': 'y_true',
                            'sigma_pred_label': 'y_pred',
                            'sigma_pred_err_label': 'y_pred_std',
                            'mass_frac_water_label': 'mass_frac',
                            'save_fig_size': figure_size,
                            'save_fig': save_fig,
                            'fig_extension': fig_extension,
                            'fig_name': os.path.join(path_dict['sim_image_dir'], fig_name) + '.' + fig_extension,
                            'want_multiple_cosolvents': False,
                            'cross_validation_training_info_stored': cross_validation_training_info_stored,
                            }
            output_path: [str]
                path to output
            save_fig: [logical, default=True]
                True if you want to save the figure
            fig_name_prefix: [str]
                string with figure name prefix to store
        OUTPUTS:
            
        '''
        ## STORING AND DIRECTORIES
        database_list = database_dict.keys()
        ## ADDING THE PARITY PLOT DICTIONARY
        parity_plot_inputs['sigma_act_label'] = 'y_true'
        parity_plot_inputs['sigma_pred_label'] = 'y_pred'
        parity_plot_inputs['sigma_pred_err_label'] = 'y_pred_std'
        parity_plot_inputs['mass_frac_water_label'] = 'mass_frac'
        parity_plot_inputs['want_multiple_cosolvents'] = False
        
        ## LOOPING THROUGH THE DATABASE LIST
        for idx, desired_database in enumerate(database_list):
            ## DEFINING PREDICTED VALUES
            stored_predicted_value_list = stored_predicted_value_list_storage[idx]
            ## CREATING DATAFRAME
            df = pd.DataFrame( stored_predicted_value_list )
            ## CREATING A CROSS VALIDATION TRAINING INFO
            cross_validation_training_info_stored={
                    'last_one': True,
                    'data': [],
                    }
            ###############################
            ### ADDING RMSE INFORMATION ###
            ###############################
            
            ## LOOPING THROUGH EACH SOLUTE
            rmse_dict = compute_df_metric_per_column( df = df,
                                                      column_name = 'solute',
                                                      y_true_label = 'y_true',
                                                      y_pred_label = 'y_pred',
                                                      current_metric = "rmse",
                                                      )

            ## LOOPING THROUGH DICTIONARY AND STORING
            for each_key in rmse_dict:
                cross_validation_training_info_stored['data'].append(
                        {'test_set_variables': each_key,
                         'test_rmse': rmse_dict[each_key]
                         }
                        )
                
            ## DEFINING FIGURE NAME
            fig_name = figure_name_list[idx]
            ## ADDING TO PARITY PLOT INPUTS
            parity_plot_inputs['fig_name'] = os.path.join(output_path, fig_name_prefix + '_' + fig_name) + '.' + parity_plot_inputs['fig_extension']
            parity_plot_inputs['cross_validation_training_info_stored'] = cross_validation_training_info_stored
                
            ## PLOTTING FIGURES
            fig, ax = trained_model.plot_test_set_parity(stored_predicted_value_list = stored_predicted_value_list, 
                                                         parity_plot_inputs = parity_plot_inputs)
        
        return


########################################################
### SAMPLING CHUNKS 
########################################################
### FUNCTION TO PLOT SAMPLING TIME CHUNKS
def publish_plot_sampling_time_chunks( sampling_time_chunks,
                                       figure_details,
                                       ns_per_frame = 10/1000.,
                                       ):
    '''
    The purpose of this function is to plot the sampling time chunks
    INPUTS:
        sampling_time_chunks: [obj]
            object containing sampling time chunks
        figure_details: [dict]
            dictionary containing all figure details
        ns_per_frame: [float]
            number of nanoseconds per frame
    OUTPUTS:
        fig, ax: figure and axis for the plot
    '''
    ## DEFINING FIGURE
    fig, ax = plotter.create_fig_based_on_cm( figure_details['figsize'] )
    ## FIXING PLOT FIGURE AND TICKS
    fix_figure_ticks_and_lims(ax = ax,
                              **figure_details['figure_limits']
                              )
    
    ## SETTING AXIS LABELS
    ax.set_xlabel("Simulation trajectory blocks")
    ax.set_ylabel("RMSE")
    
    ## DEFINING PLOTTING VARIABLES
    split_time_storage = np.array(sampling_time_chunks.split_time_storage)
    rmse_storage = sampling_time_chunks.rmse_storage
    
    ## NS PER FRAME
    split_time_storage_ns = split_time_storage * ns_per_frame
    
    ## DEFINING X AND Y
    x = [ "%d-%d ns"%(each_split[0], each_split[1]) for each_split in split_time_storage_ns]
    y = np.array(rmse_storage)
    
    ## ADDING Y LINE AT 0.10
    ax.axhline( y = 0.10, linestyle = '--', color = 'k', linewidth=2)
    
    ## PLOTTING BAR PLOT
    ax.bar(x = x, height =y , color='k', width=0.8, alpha=0.8)
    
    ## ROTATING X LABEL
    ax.set_xticklabels(x, rotation=90, ha='center')
    
    ## SHOW TIGHT LAYOUT
    fig.tight_layout()
    
    
    return fig, ax

### FUNCTION TO COMPUTE DETAILS FOR CROSS VALIDATION STORAGE
def compute_cross_valid_statistics(cross_validation_storage, 
                                   desired_stats = ['slope', 'rmse', 'pearson_r'] ):
    '''
    The purpose of this function is to extract cross validation storage 
    information. In particular, it will extract test set RMSE, slope, and 
    pearson's r. 
    INPUTS:
        cross_validation_storage: [dict]
            dictionary containing all details of cross validation details.
        desired_stats: [list]
            list of desired stats
    OUTPUTS:
        output_stats: [dict]
            output statistics dictionary:
                'rmse': root mean squared error of the test set
                'slope': slope of the test set
                'pearson_r': pearson's correlation between predicted and experiments
    '''

    ## DEFINING DATA FRAME
    dataframe = cross_validation_storage['dataframe']
    ## DEFINING CROSS VALIDATION ANME
    cross_valid_name = cross_validation_storage['cross_validation_training_info']['cross_validation_name']
    cross_valid_test_var = cross_validation_storage['cross_validation_training_info']['test_set_variables']
    ## EXTRACTING PREDICTED AND EXP TEST SET
    test_set_index = dataframe[cross_valid_name] == cross_valid_test_var
    test_set_exp = dataframe[test_set_index]['y_true']
    test_set_pred = dataframe[test_set_index]['y_pred']
    

    current_metrics = metrics( y_fit = test_set_pred, y_act = test_set_exp )
    current_rmse = current_metrics[1]
    current_slope = current_metrics[4]
    pearson_r = pearsonr( x = test_set_exp, y = test_set_pred )[0]
    
    ## DEFINING OUTPUT DICT
    output_stats = {}
    
    ## DEFINING STORAGE
    if 'rmse' in desired_stats:
        output_stats['rmse'] = current_rmse
    if 'slope' in desired_stats:
        output_stats['slope'] = current_slope
    if 'pearson_r' in desired_stats:
        output_stats['pearson_r'] = pearson_r
    
    return output_stats

### FUNCTION TO EXTRACT NAMES AND TEST RMSE
def extract_cross_valid_rmse(cross_valid_storage, 
                             desired_name=None,
                             desired_stats = ['slope', 'rmse', 'pearson_r']):
    '''
    The purpose of this function is to extract details from the model storage
    INPUTS:
        cross_valid_results: [dict]
            cross validation storage dictionary
        desired_name: [str, default=None]
            desired name of the row
        desired_stats: [list]
            list of desired stats
    OUTPUTS:
        df: [pandas dataframe]
            dataframe containing extracted information
    '''
    ## CREATING EMPTY DICTIONARY
    extracted_dict = {}
    ## LOOPING THROUGH EACH KEY
    for each_result in cross_valid_storage:
        ## COMPUTING STATS FOR EACH
        each_stats = compute_cross_valid_statistics( each_result )
        ## DEFINING EACH_KEY
        each_key = each_result['cross_validation_training_info']['test_set_variables']
        ## CREATING ENTRY
        extracted_dict[each_key] = {}
        ## STORING
        if desired_name is None:
            key_name='test_'
        else:
            key_name = desired_name
            
        ## LOOPING THROUGH EACH KEY
        key_list  = { each_key: each_key + '_' +  key_name for each_key in each_stats.keys() if each_key in desired_stats }
        for current_key in key_list.keys():
            extracted_dict[each_key][key_list[current_key]] = each_stats[current_key]
            # each_result['cross_validation_training_info']['test_rmse']
    
    ## CREATING A DATAFRAME
    df = pd.DataFrame(extracted_dict)
    return df

### FUNCTION TO EXTRACT CROSS VALIDATION OF NN
def extract_cross_valid_nn( cross_valid_storage, 
                            desired_stats = ['slope', 'rmse', 'pearson_r'],
                            desired_name = None ):
    '''
    The purpose of this function is to extract the cross validation nn
    INPUTS:
        cross_valid_results: [dict]
            dictionary object containing all prediction statistics
        desired_name: [str, default=None]
            desired name of the row
        NOTE: This will only store the first entry of neuron list
    OUTPUTS:
        df: [pandas dataframe]
            dataframe containing extracted information
    '''
    ## CREATING EMPTY DICTIONARY
    extracted_dict = {}
    ## LOOPING THROUGH EACH KEY
    for each_key in cross_valid_storage:
        ## CREATING ENTRY
        extracted_dict[each_key] = {}
        ## GETTING STATISTICS FOR EACH
        key_name = desired_name
        ## LOOPING THROUGH EACH KEY
        key_list  = { each_key: each_key + '_' +  key_name for each_key in desired_stats }
        ## LOOPING THROUGH EACH STAT
        for each_stat in desired_stats:
            extracted_dict[each_key][key_list[each_stat]] = cross_valid_storage[each_key]['predict_stats'][0][each_stat]
    ## CREATING A DATAFRAME
    df = pd.DataFrame(extracted_dict)
    return df
    
    
### FUNCTION TO PLOT DOUBLE COLUMN
def plot_double_col(csv_comparison,
                       figure_size,):
    '''
    The purpose of this function is to plot the double column for slope and rmse
    INPUTS:
        csv_comparison: [dataframe]
            dataframe containing information about the slope / rmse of each mdoel
        figure_size: [tuple]
            figure size in centimeters
    OUTPUTS:
        
    '''
    ## IMPORTING MATPLOTLIB
    import matplotlib.pyplot as plt
    ## DEFINING BAR WIDTH
    bar_width = 0.4 # 0.35
    ## DEFINING Y POSITIONS
    y_pos = np.arange(csv_comparison.shape[0])
    ## REVERSING
    # y_pos = y_pos[::-1]
    ## FINDING FIGURE SIZE
    figsize=plotter.cm2inch( *figure_size )
    ## DOUBLE BAR PLOT
    fig, ax = plt.subplots(figsize = figsize )
    
    ## ADDING SECCOND AXIS
    ax2 = ax.twiny()
    
    ## DEFINING X TICKS
    x_bot_ticks = np.arange(0,0.6+0.1, 0.1)
    x_top_ticks = np.arange(0,1+0.25, 0.25)

    ## DEFINING RECTS 2
    comparison_var = 'rmse'
    rects1 = ax.barh(y_pos-bar_width/2, csv_comparison[comparison_var], height=bar_width, align='center', color = 'k', label = comparison_var )
    
    ## DEFINING DESIRED COMPARISON
    comparison_var = 'slope'
    rects2 = ax2.barh(y_pos+bar_width/2, csv_comparison[comparison_var], height=bar_width, align='center', color = 'gray', label = comparison_var )
    
    ## defining labels
    labels = np.insert(np.array(csv_comparison['model']), 0, '', )
    
    ## ADDING Y TICKS
    ax.set_yticklabels(labels = labels) # y_pos, 
    
    ## SETTING X TICKS
    ax.set_xticks(x_bot_ticks)
    ax2.set_xticks(x_top_ticks)
    
    ## CHANGING TOP COLOR
    ax2.xaxis.label.set_color('gray')
    ax2.tick_params(axis='x', colors='gray')
    ax2.spines['top'].set_color('gray')
    
    ## ADDING RIGHT Y TICKS
    # ax.yaxis.set_label_position("right")
    # ax.tick_params(axis='y', which='both', labelleft='on', labelright='on')
    # ax2.tick_params(axis='y', which='both', labelleft='on', labelright='off')
    
    ## SETTING LIMIS
    ax.set_xlim( [np.min(x_bot_ticks), np.max(x_bot_ticks)] )
    ax2.set_xlim( [np.min(x_top_ticks), np.max(x_top_ticks)] )
    
    ## SETTING LABELS
    ax.set_xlabel('rmse')
    ax2.set_xlabel('slope')
    return fig

### FUNCTION TO LOAD INSTANCES BASED ON PICKLE NAME
def load_instances_based_on_pickle(pickle_name,
                                   path_combined_database,
                                   ):
    '''
    The purpose of this function is to load instances based on pickle that was 
    trained. 
    INPUTS:

    OUTPUTS:
    '''
    ## DEFINING COMBINED NAME
    combined_name_info = read_combined_name( unique_name = pickle_name,
                                            reading_type = 'instances',
                                            )
    ## DEFINING COMBINED DATABASAE PATH
    database_path = os.path.join( path_combined_database, combined_name_info['data_type'])  # None # Since None, we will find them!        
    
    ## UPDATING REPRESATION INPUTS
    representation_inputs = extract_representation_inputs( representation_type = combined_name_info['representation_type'], 
                                                          representation_inputs = combined_name_info['representation_inputs'].split('_') )
        
    ## RUNNING COMBINED INSTANCES
    instances = combine_instances(
                     solute_list = combined_name_info['solute_list'],
                     representation_type = combined_name_info['representation_type'],
                     representation_inputs = representation_inputs,
                     solvent_list = combined_name_info['solvent_list'], 
                     mass_frac_data = combined_name_info['mass_frac_data'], 
                     verbose = True,
                     database_path = database_path,
                     class_file_path = class_file_path,
                     combined_database_path = combined_database_path,
                     data_type = combined_name_info['data_type'],
                     enable_pickle = True, # True if you want pickle on
                     )
    
    return instances


### FUNCTION TO PLOT FOR A GIVEN INSTANCE
def publish_voxel_image(instances, 
                        instance_name,
                        specific_partition = 0,
                        plot_voxel_split_inputs = {
                                'alpha': 0.3,
                                'want_renormalize': True,
                                }
                        ):
    '''
    The purpose of this function is to plot the voxel image and publish it. 
    INPUTS:
        instances: [obj]
            class object containing instances information
        instance_name: [str]
            name of the instance you want to extract from instance class
        specific_partition: [int]
            specific partition that you are interested in
        plot_voxel_split_inputs: [dict]
            dictionary of plot voxel inputs
    '''
    ## FINDING INSTANCE
    index_instance = instances.instance_names.index( specific_instance )
    
    ## FINDING INSTANCE
    index_instance = instances.instance_names.index( specific_instance )
    
    ## DEFINING DATA
    rgb_data = instances.x_data[index_instance][specific_partition]
    
    ## PLOTTING
    fig, ax = plotter.plot_voxel_split(rgb_data, **plot_voxel_split_inputs)
    
    return fig, ax

### FUNCTION TO TURN OFF LABELS
def turn_ax_labels_off(ax):
    ''' This function turns off axis labels '''
    ## GETTING LABELS
    labels = [item.get_text() for item in ax.get_xticklabels()]
    ## GETTING STRING OF LABELS
    empty_string_labels = ['']*len(labels)
    ## SETTING ALL LABELS TO NOTHING
    ax.set_xticklabels(empty_string_labels)
    ax.set_yticklabels(empty_string_labels)
    ax.set_zticklabels(empty_string_labels)
    ## SETTING LABELS TO NOTHING
    ax.set(xlabel='', ylabel='', zlabel='')
    return

#%%
## MAIN FUNCTION
if __name__ == "__main__":

    ############################################################################
    ### FIGURE 2A -- Human descriptors
    ############################################################################

    ## DEFINING FIGURE SIZE
    figure_size=( 18.542/3, 18.542/3 )
    # ( 18.542/2, 18.542/2 )
    
    ## DEFINING FIGURE NAME
    fig_name = r"2A_Predicted_Human_Descriptors"
    
    ## DEFINING PATH TO REGRESSION FILE
    path_md_pred_regression = path_dict['path_md_descriptors_regression']

    ## USING PANDAS TO READ FILE    
    csv_file = pd.read_csv( path_md_pred_regression )
     
    ## GETTING DATA ONLY FOR DIO, GVL, AND THF
    dataframe = csv_file
    
    ## PLOTTING 
    plot_parity_publication_single_solvent_system( dataframe = dataframe,
                                                   fig_name = os.path.join(path_image_dir, fig_name) + '.' + fig_extension,
                                                   mass_frac_water_label = 'mass_frac_water',
                                                   sigma_act_label = 'sigma_label',
                                                   sigma_pred_label = 'sigma_pred',
                                                   save_fig_size = figure_size, # (16.8/3, 16.8/3),
                                                   fig_extension = fig_extension, # fig_extension
                                                   save_fig = save_fig)
    
    
    #%%
    ############################################################################
    ### FIGURE 2B -- Human descriptors with neural network model
    ############################################################################
    
    ## DEFINING FIGURE NAME
    figname = "2B_md_descriptor_testing"
    ## DEFINING NN MODEL
    nn_model = nn_descriptors_model( **NN_DESCRIPTOR_MODEL_INPUTS )
 
    #######################
    ### PRINTING FIGURE ###
    #######################
    ## DEFINING FIG NAME
    fig_name = os.path.join(path_image_dir, figname) + '.' + fig_extension
    
    ## DEFINING INPUT TO PARITY PLOT
    parity_plot_inputs = {
            'save_fig_size' : ( 18.542/2, 18.542/2 ),
            'fig_name' : fig_name,
            'save_fig' : save_fig,
            }
    
    ## GENERAITNG PARITY PLOT
    nn_model.plot_parity(parity_plot_inputs = parity_plot_inputs,
                         )
    
    ## PRINTING SLOPE AND RMSE
    print("Slope: %.15f"%(nn_model.predict_stats[0]['slope']) )
    print("RMSE: %.15f"%(nn_model.predict_stats[0]['rmse']) )

    #%%
    ############################################################################
    ### FIGURE 3: Different voxel representations
    ############################################################################

    ## DEFINING DATABASE NAME
    database_name=r"20_20_20_20ns_oxy_3chan"
    
    ## FIGURE 2(part 3) VOXELATION AFTER 4 NS
    pickle_location= os.path.join(path_dict['combined_database_path'])
    pickle_name= database_name + r"-split_avg_nonorm-10-CEL_ETBE_FRU_LGA_PDO_XYL_tBuOH-DIO_GVL_THF-10_25_50_75"
 
    ## LOADING INSTANCES
    instances = load_instances_based_on_pickle(pickle_name = pickle_name ,
                                               path_combined_database = path_dict['combined_database_path'])
    
    ## DEFINING SPECIFIC PARTITION
    specific_partition = 0
    
    ## DEFINING INSTANCES LIST
    instances_list = [
            "XYL_403.15_DIO_10",
            "XYL_403.15_DIO_25",
            "XYL_403.15_DIO_50",
            "XYL_403.15_DIO_75",
            "tBuOH_363.15_DIO_10",
            "tBuOH_363.15_DIO_25",
            "tBuOH_363.15_DIO_50",
            "tBuOH_363.15_DIO_75",
            "XYL_403.15_GVL_10",
            "XYL_403.15_GVL_75",
            "tBuOH_363.15_GVL_10",
            "tBuOH_363.15_GVL_75",
            "XYL_403.15_THF_10",
            "XYL_403.15_THF_75",
            "tBuOH_363.15_THF_10",
            "tBuOH_363.15_THF_75",
            ]
    
    ## LOOPING
    for specific_instance in instances_list:
    
        ## DEFINING FIGURE NAME
        fig_name = specific_instance + '_avg_2ns_frame'
        
        ## FINDING INSTANCE
        index_instance = instances.instance_names.index( specific_instance )
        
        ## DEFINING DATA
        rgb_data = instances.x_data[index_instance][specific_partition]
        
        ## PLOTTING
        fig, ax = plotter.plot_voxel_split(rgb_data,
                                   alpha=0.3,
                                   want_renormalize = True)
        
        ## STORING FIGURE
        store_figure( fig = fig, 
                      path = os.path.join(path_image_dir, fig_name + '_' + specific_instance),
                      save_fig = save_fig,
                      fig_extension = fig_extension,
                      dpi = 300, # Lower dpi for storing such a large picture
                     )
    
    #%%
    ###################################
    ### PRINTING SPECIFIC INSTANCES ###
    ###################################
    
    ## PICKLE LOADING
    from extraction_scripts import load_pickle
    
    
    ## DEFINING DATABASE NAME
    database_name=r"20_20_20_20ns_oxy_3chan"
    ## DEFINING SPECIFIC INSTANCE
    specific_instance = "XYL_403.15_DIO_10"
    ## DEFINING PATH
    database_path = os.path.join( path_dict['database_path'], database_name, specific_instance )
    ## LOADING PICKLE
    rgb_data = load_pickle(database_path)
    #%%
    ## DEFINING FRAME
    frame = 1
    
    ## PLOTTING
    fig, ax = plotter.plot_voxel_split(rgb_data,
                               frame = frame,
                               alpha=0.10,
                               want_renormalize = False) # Doesn't matter to renormalize in this case
    
    ## STORING FIGURE
    store_figure( fig = fig, 
                  path = os.path.join(path_image_dir, specific_instance + '_' + str(frame)),
                  save_fig = save_fig,
                  fig_extension = fig_extension,
                  dpi = 300, # Lower dpi for storing such a large picture
                 )
    
    
    

    #%%
    ############################################################################
    ## FIGURE 4C: SolventNet Predictions
    ############################################################################
    
    ## DEFINING VARIABLES
    results_file_path = simulation_path_dicts['3D_CNN_Training_All_Solvents_Solvent_Net']
    
    ## DEFINING FIGURE SIZE
    figure_size=FIGURE_SIZES_CM_SI['2_col']/2
    # ( 18.542/3, 18.542/3 )
    # ( 18.542/2, 18.542/2 ) <-- manuscript
    
    ## DEFINING FIGURE NAME
    fig_name = r"4C_Predicted_Solvent_Net"

    ## DEFINING PICKLE FILE
    results_pickle_file = r"model.results" # THF
    
    ## DEFINING FULL PATH
    results_full_path = os.path.join( results_file_path, results_pickle_file )
    
    ## DEFINING ANALYSIS
    analysis = pd.read_pickle( results_full_path  )[0]
    
    ## DEFINING DATAFRAME
    df = analysis.dataframe
    
    ## PLOTTING PARITY
    plot_parity_publication_single_solvent_system( dataframe = df,
                                                   fig_name = os.path.join(path_image_dir, fig_name) + '.' + fig_extension,
                                                   mass_frac_water_label = 'mass_frac',
                                                   sigma_act_label = 'y_true',
                                                   sigma_pred_label = 'y_pred',
                                                   sigma_pred_err_label = 'y_pred_std',
                                                   fig_extension = fig_extension,
                                                   save_fig_size = figure_size,
                                                   save_fig = save_fig)
    
    #%%
    

    ## LOOPING THROUGH MODELS    
    for model_index in range(len(analysis.model_y_pred_raw)):
    
        ## FINDING Y PREDICTED
        y_pred_avg, y_pred_std, y_true_split = find_avg_std_predictions(instance_names = analysis.instance_names, 
                                                                        y_pred = analysis.model_y_pred_raw[model_index],
                                                                        y_true = analysis.y_true_raw)
        
        ## CREATING DATAFRAME
        dataframe = create_dataframe(instance_dict = analysis.instance_dict,
                                     y_true = analysis.y_true,
                                     y_pred = y_pred_avg,
                                     y_pred_std = y_pred_std)
        
        
        ## PLOTTING PARITY
        plot_parity_publication_single_solvent_system( dataframe = dataframe,
                                                       fig_name = os.path.join(path_image_dir, fig_name) + "_mod_" + str(model_index) + '.' + fig_extension,
                                                       mass_frac_water_label = 'mass_frac',
                                                       sigma_act_label = 'y_true',
                                                       sigma_pred_label = 'y_pred',
                                                       sigma_pred_err_label = 'y_pred_std',
                                                       fig_extension = fig_extension,
                                                       save_fig_size = figure_size,
                                                       save_fig = save_fig)
    
    
    
    #%%
    ###########################################################################
    ### FIGURE 4D: BAR PLOT, NOTE YOU MAY NEED TO CHANGE PATH OF COMPARISON
    ###########################################################################
    ## MAKING BAR PLOT
    # https://pythonspot.com/matplotlib-bar-chart/
    
    ## DEFINING FULL PATH TO DATA
    path_comparison= path_dict['path_comparison']
    
    ## READING CSV
    csv_comparison = pd.read_csv(path_comparison)
    
    ## DEFINING FIGURE NAME
    fig_name = "4D_Comparison_btn_models"
    
    ## DEFINING FIGURE SIZE
    figure_size_inches = np.array([2.4, 6.5/2]) # width, height inches
    figure_size = tuple(figure_size_inches*2.54) # cm
    
    ## REVERSE CSV
    csv_comparison = csv_comparison.reindex(index=csv_comparison.index[::-1])
    
    ## PLOTTING FIGURE
    fig = plot_double_col(csv_comparison = csv_comparison,
                           figure_size = figure_size,)    
    
    ## STORING FIGURE
    store_figure( fig = fig, 
                  path = os.path.join(path_image_dir, fig_name),
                  save_fig = save_fig,
                  fig_extension = fig_extension,
                 )
    
    #%%
    
    ###########################################################################
    ### FIGURE 5A,B: CROSS VALIDATION ACROSS COSOLVENTS
    ###########################################################################
    
    ## DEFINING FIGURE SIZE
    figure_size=( 18.542/3, 18.542/3 )
    
    ## DEFINING MAIN DIRECTORY LIST
    main_dir_dict={
            # 'cosolvent': os.path.basename(simulation_path_dicts['cross_validation_paths']['SolventNet_Cosolvent']) ,
            'reactant': os.path.basename(simulation_path_dicts['cross_validation_paths']['SolventNet_Solute']) ,
            }    
    
    ## CREATING CROSS VALIDATION
    cross_valid_extracted = extract_cross_validation()

    ## DEFINING OUTPUT CSV FILE
    path_output_cross_csv = os.path.join(path_output_excel_spreadsheet, "solvent_net_cross_valid_cosolvent.csv")
    
    ## DEFINING FIGURE NAME
    fig_name = "5_solvent_net_cross_val"
    
    ## DEFININING PICKLE
    results_pickle_file = "model.results"
    # "model_fold_4.pickle"
    
    ## DEFINING MAIN DIRECTORY
    for main_dir_key in main_dir_dict:
        ## DEFINING MAIN DIRECTORY
        main_dir = main_dir_dict[main_dir_key]
        ## NEW FIGURE NAME
        current_fig_name = fig_name + '_' + main_dir_key
        
        ## DEFINING INPUTS
        cross_valid_inputs = {
                'main_dir': main_dir,
                'combined_database_path': combined_database_path,
                'class_file_path': class_file_path,
                'image_file_path': path_dict['path_image_dir'],
                'sim_path': sim_path,
                'database_path': database_path,
                'results_pickle_file': results_pickle_file,
                'verbose': True,
                }
        
        ## LOADING CROSS VALIDATION
        cross_valid_results = cross_valid_extracted.load_cross_validation(cross_valid_inputs  = cross_valid_inputs,
                                                                          pickle_path = path_dict['path_pickle'])
        
        ## DEFINING PLOTTING INPUTS
        parity_plot_inputs = \
            {
                    'save_fig_size': figure_size,
                    'save_fig': save_fig,
                    'fig_name': os.path.join(path_image_dir, current_fig_name) + '.' + fig_extension,
                    'fig_extension': fig_extension,
                    'x_lims': (-1.5, 3, 1 ),
                    'y_lims': (-1.5, 3, 1 ),
                    }
        ## PLOTTTING FIGURE
        fig, ax = cross_valid_extracted.plot_parity_plot(cross_valid_results = cross_valid_results,
                                                         parity_plot_inputs = parity_plot_inputs,
                                                         want_combined_plot = True)
        
        

    #%%
    ###########################################################################
    ### FIGURE 6: PREDICTIONS FOR NEW SOLVENT SYSTEMS
    ###########################################################################
    ## FINDING SIM PATH
    sim_path = path_dict['sim_path']
    
    ## DEFINING PATH TO MAIN DIRECTORY
    path_main_dir = simulation_path_dicts['3D_CNN_Training_All_Solvents_Solvent_Net']
    
    ## DEFINING MAIN DIRECTORY
    main_dir = os.path.basename(path_main_dir)
    
    ## DEFINING MODEL WEIGHTS ARRAY
    model_weights_list = [ 
            "model_fold_0.hdf5",
            "model_fold_1.hdf5",
            "model_fold_2.hdf5",
            "model_fold_3.hdf5",
            "model_fold_4.hdf5",
            ]
    
#    ## DEFINING MODEL WEIGHTS
#    model_weights = "model.hdf5"
    
    ## DEFINING FULL PATH TO MODEL
    path_model = [ os.path.join(path_main_dir, model_weights) for model_weights in model_weights_list ] 
    
    ## DEFINING INPUTS FOR PREDICTED MODEL
    inputs_predicted_model = {
            'path_model': path_model,
            'verbose': True,
            }
    ## LOADING MODEL
    trained_model = predict_with_trained_model( **inputs_predicted_model )
    ## FIGURE 6A, B, C
## DEFINING FIG NAME
    #%%
    ## WANT RECREATE PICKLE
    want_repickle = True
    
    ## DEFINING PREDICTIVE MODEL
    predict_model = publish_predict_model()

    ## DEFINING DATABASE
    test_database_basename = main_dir.split('-')[0] + '_' #  '20_20_20_20ns_oxy_3chan_'
    
    ## GETTING DATABASE DICT
    database_dict = get_test_pred_test_database_dict(test_database_basename = test_database_basename)
    
    ## DEFINING DESIRED
    desired_database = "DMSO"
    
    ''' DEBUGGING PREDICTING TEST TOOL
    predicted_storage = trained_model.predict_test_set(path_test_database = database_dict[desired_database]['path_database'],
                                                       num_partitions = 2, )
    '''
    
    #%%
    ## LOOPING AND PREDICTING
    stored_predicted_value_list_storage, figure_name_list = predict_model.loop_database_and_predict(trained_model = trained_model,
                                                                                                    main_dir = main_dir,
                                                                                                    path_pickle = path_dict['path_pickle'],
                                                                                                    database_dict = database_dict,
                                                                                                    want_repickle = want_repickle,
                                                                                                    num_partitions = 2,
                                                                                                    )
    #%%
    ## DEFINING FIGURE SIZE
    figure_size=( 18.542/3, 18.542/3 )
    
    ## DEFINING PARITY PLOT INPUTS    
    parity_plot_inputs = \
        {
                'save_fig_size': figure_size,
                'save_fig': save_fig,
                'fig_extension': fig_extension,
                }       
    predict_model.plot_parity_plot(stored_predicted_value_list_storage = stored_predicted_value_list_storage,
                                   figure_name_list = figure_name_list,
                                   parity_plot_inputs = parity_plot_inputs,
                                   output_path = path_dict['sim_image_dir'],
                                   )
    #%%
    
    ## TABLE SX: Table on predictive model and csv datas
    
    ## FUNCTION TO GENERATE CSV STATS
    predict_model.generate_csv_stats(stored_predicted_value_list_storage = stored_predicted_value_list_storage,
                                     database_dict = database_dict,
                                     figure_name_list = figure_name_list,
                                     path_output_excel_spreadsheet =  path_output_excel_spreadsheet,
                                     desired_metrics = [  "slope","rmse", "pearson" ] )
        
    #%%
    
    ###########################################################################
    ### FIGURE S1A: Sampling time increments varying training sizes
    ###########################################################################
        
    ## DEFINING PATH TO SIM
    path_sim_dir = simulation_path_dicts['Increment_varying_training']
    
    ## DEFINING INPUTS
    inputs={
            'path_sim_dir': path_sim_dir,
            'results_pickle_file': "model.results",
            'image_file_path': None,
            'print_csv': False,
            'save_fig' : False,
            'want_fig' : False,
            }
    
    ## RUNNING ANALYSIS
    read_sampling_times = read_sampling_time_increments_with_varying_training_sizes( **inputs )
    
    ## DEFINING FIGURE NAME
    fig_name = "SI_1A_sampling_time_varying_training_size_updated"
    
    ## DEFINING FIGURE DETAILS
    figure_details = {
            'figsize': FIGURE_SIZES_CM_SI['1_col'],
            'figure_limits':{
                'x_ticks': (0, 10, 1),
                'y_ticks': (0, 0.40, 0.05),
                'x_lims' : (0, 10.5),
                'y_lims' : (0, 0.40),}
            }

    ## PLOTTING SAMPLING TIME INCREMENTS
    fig, ax = publish_plot_sampling_time_increments( read_sampling_times = read_sampling_times,
                                                     figure_details = figure_details,
                                                     amount_ns_per_partition = 10,
                                                     )
    ## STORING FIGURE
    store_figure( fig = fig, 
                  path = os.path.join(path_image_dir, fig_name),
                  save_fig = save_fig,
                  fig_extension = fig_extension,
                 )
    #%%
    ###########################################################################
    ### FIGURE S1B: Time invariance by taking chunks of time
    ###########################################################################
    
    ## DEFINING PATH TO SIM
    path_sim_dir = simulation_path_dicts['Sampling_chunks_training']
    
    ## DEFINING INPUTS
    inputs={
            'path_sim_dir': path_sim_dir,
            'results_pickle_file': "model.results",
            'image_file_path': None,
            'print_csv': False,
            'save_fig' : False,
            'want_fig' : False,
            }
    ## RUNNING ANALYSIS
    sampling_time_chunks = read_sampling_time_chunks( **inputs )
    
    ## EXTRACTING MAIN MANUSCRIPT RMSE
    path_main_manuscript = simulation_path_dicts['3D_CNN_Training_All_Solvents_Solvent_Net']
    results_pickle_file = r"model.results" # THF
    
    ## DEFINING FULL PATH
    results_full_path = os.path.join( path_main_manuscript, results_pickle_file )
    
    ## DEFINING ANALYSIS
    analysis = pd.read_pickle( results_full_path  )[0]
    rmse = analysis.accuracy_dict['rmse']
    
    ## STORING IN RMSE STORAGE TO BEGINING OF LIST
    index = sampling_time_chunks.split_time_storage.index( [0,2000])
    ## STORING INTO RMSE
    sampling_time_chunks.rmse_storage[index] = rmse
    ## SORTING
    sorted_index = np.argsort( sampling_time_chunks.split_time_storage, axis = 0, )[:,0]
    ## RESORTING ARRAYS
    sampling_time_chunks.split_time_storage = [ sampling_time_chunks.split_time_storage[each_index] for each_index in sorted_index]
    sampling_time_chunks.rmse_storage = [ sampling_time_chunks.rmse_storage[each_index] for each_index in sorted_index]

    ## DEFINING FIGURENAME
    fig_name = "SI_1B_sampling_time_varying_chunks"
    
    ## DEFINING FIGURE DETAILS
    figure_details = {
            'figsize': FIGURE_SIZES_CM_SI['1_col'],
            'figure_limits':{
                'y_ticks': (0, 0.20, 0.05),
                'y_lims' : (0, 0.20),}
            }
    
    ## PLOTTING SAMPLING TIME INCREMENTS
    fig, ax = publish_plot_sampling_time_chunks( sampling_time_chunks = sampling_time_chunks,
                                                     figure_details = figure_details,
                                                     ns_per_frame = 10/1000.,
                                                     )
    
    ## STORING FIGURE
    store_figure( fig = fig, 
                  path = os.path.join(path_image_dir, fig_name),
                  save_fig = save_fig,
                  fig_extension = fig_extension,
                 )
    
    #%%
    ###########################################################################
    ### SI FIGURE S2: Parity plot for VoxNet and ORION
    ###########################################################################
    
    ## DEFINING FIGNAME
    figname = "SI_2_Parity_"
    
    ## DEFINING RESULTS PICKLE
    results_pickle_file = r"model.results" # THF
    
    ## DEFINING NETWORKS
    networks = [ "voxnet", "orion" ]
    
    ## DEFINING FIGURE SIZE
    figure_size= FIGURE_SIZES_CM_SI['2_col']/len(networks)
    
    ## LOOPING THROUGH EACH NETWORK
    for each_network in networks:
        current_figname = figname + each_network
        ## DEFINING VARIABLES
        results_file_path =simulation_path_dicts['3D_CNN_Training_All_Solvents_' + each_network]
        ## DEFINING FULL PATH
        results_full_path = os.path.join( results_file_path, results_pickle_file )
        ## DEFINING ANALYSIS
        analysis = pd.read_pickle( results_full_path  )[0]
        
        df = analysis.dataframe # dataframe.set_index('solute')
        
        ## PLOTTING PARITY
        plot_parity_publication_single_solvent_system( dataframe = df,
                                                      fig_name = os.path.join(path_image_dir, current_figname) + '.' + fig_extension,
                                                       mass_frac_water_label = 'mass_frac',
                                                       sigma_act_label = 'y_true',
                                                       sigma_pred_label = 'y_pred',
                                                       sigma_pred_err_label = 'y_pred_std',
                                                       fig_extension = fig_extension, # fig_extension
                                                       save_fig_size = figure_size, # (16.8/3, 16.8/3),
                                                       save_fig = save_fig)
    #%%
    ###########################################################################
    ### SI FIGURE S3: LEARNING CURVE
    ###########################################################################
    ## DEFINING FIGURE SIZE
    figure_size=( 8.3, 8.3 )
    
    ## DEFINING PICKLE
    results_pickle = "model.results"
    
    ## DEFINING FIGURE NAME
    fig_name = "SI_2_Learning_curve"
    
    ## CREATING LEARNING CURVE CLASS
    learning_curve = publish_plot_learning_curve(figure_size = figure_size)
    
    ## DEFINING DICTIONARY
    simulation_dict = {
            'voxnet': simulation_path_dicts['3D_CNN_Training_All_Solvents_voxnet'],
            'orion': simulation_path_dicts['3D_CNN_Training_All_Solvents_orion'],
            'solvent_net': simulation_path_dicts['3D_CNN_Training_All_Solvents_Solvent_Net'],
            }
    
    # for network_type in ['voxnet','orion','solvent_net']:
    for network_type in simulation_dict.keys():
        
        ## DEFINING PATH
        sim_path = simulation_dict[network_type]
        
        ## DEFINING PICKLE PATH
        path_pickle = os.path.join(sim_path, results_pickle)
        
        ## LOADING PICKLE
        analysis = pd.read_pickle( path_pickle  )[0]
        
        ## PLOTTING LEARNING CURVE
        learning_curve.plot_each_loss_per_color( history = analysis.history,
                                                 network_type = network_type)
        
    ## FINALIZING LEARNING CURVE
    learning_curve.finalize_image()
    
    ## STORING FIGURE
    store_figure( fig = learning_curve.fig, 
                  path = os.path.join(path_image_dir, fig_name),
                  save_fig = save_fig,
                  fig_extension = fig_extension,
                 )
    
    
    #%%
    
    ###########################################################################
    ### SI: LEARNING CURVE 
    ###########################################################################
    
    ## DEFINING FIGURE SIZE
    figure_size=( 8.3, 8.3 )
    
    ## DEFINING FIGURE NAME
    fig_name = "SI_2_learning_curve_solventnet"
    
    

    
    ## DEFINING PICKLE
    results_pickle = "model.results"
    
    ## DEFINING SIMULATION LIST
    simulation_dict = {
            # 'voxnet': simulation_path_dicts['3D_CNN_Training_All_Solvents_voxnet'],
            # 'orion': simulation_path_dicts['3D_CNN_Training_All_Solvents_orion'],
            'solvent_net': simulation_path_dicts['3D_CNN_Training_All_Solvents_Solvent_Net'],
            }
    
    ## DEFINING DICT FOR MODELS
    model_dict = {
            'Fold 1': "model_fold_0.pickle",
            'Fold 2': "model_fold_1.pickle",
            'Fold 3': "model_fold_2.pickle",
            'Fold 4': "model_fold_3.pickle",
            'Fold 5': "model_fold_4.pickle",
            }
    
    
    ## DEFINING COLOR
    colors_list = [ 'b', 'r', 'g', 'cyan', 'k']
    
    ## LOOPING
    for network_type in simulation_dict.keys():
        ## DEFINING PATH
        sim_path = simulation_dict[network_type]
        
        ## CREATING LEARNING CURVE CLASS
        learning_curve = publish_plot_learning_curve(figure_size = figure_size)
        
        ## LOOPING THROUGH MODELS
        for idx, pickle_key in enumerate(model_dict):
            ## DEFINING PICKLE PATH
            path_pickle = os.path.join(sim_path, model_dict[pickle_key])
    
            ## LOADING PICKLE
            history, time_hms, y_pred, y_test, indices_dict = pd.read_pickle( path_pickle )
            
            ## DEFINING COLOR
            color = colors_list[idx]
            
            ## PLOTTING LEARNING CURVE
            learning_curve.plot_each_loss_per_color( history = history,
                                                     color = color
                                                     )
        ## FINALIZING LEARNING CURVE
        learning_curve.finalize_image()
    
        ## STORING FIGURE
        store_figure( fig = learning_curve.fig, 
                      path = os.path.join(path_image_dir, fig_name),
                      save_fig = save_fig,
                      fig_extension = fig_extension,
                     )
    

    #%%
    
    ###########################################################################
    ### TABLE S2: TABLE TO GET SLOPE AND RMSE OF EACH SOLVENT SYSTEM
    ###########################################################################
    
    ## DEFINING RESULTS PICKLE
    results_pickle_file = r"model.results" # DIO 
    
    ## DEFINING LIST OF DIRECTORIES TO LOOK IN (3D CNNS)
    list_of_directories = [
            r'190725-newrep_20_20_20_20ns_oxy_3chan', # SOLVENTNET
            r'190802-parity_diff_systems_20_20_20_20ns_oxy_3chan', # VOXNET, ORION
            r'190807-individual_solvents_20_20_20_20ns_oxy_3chan', # INDIVIDUAL SOLVENTS
            r'190802-parity_vgg16_32_32_32_20ns_oxy_3chan_firstwith10', # VGG16
            ]
    
    ## CREATING DATAFRAME
    df = generate_dataframe_slope_rmse(sim_path = path_dict['sim_path'],
                                       list_of_directories = list_of_directories,
                                       results_pickle_file = 'model.results',
                                       )
    
    ## DEFINING OUTPUT CSV FILE
    path_output_cross_csv = os.path.join(path_output_excel_spreadsheet, "SI_table_slope_rmse.csv")
    
    ################
    ### NN model ###
    ################
    nn_descriptors_model_inputs = { **NN_DESCRIPTOR_MODEL_INPUTS }
    nn_descriptors_model_inputs['path_sim'] = r"R:\scratch\3d_cnn_project\simulations\FINAL\2B_md_descriptor_testing"    
    ## DEFINING NN MODEL
    nn_model = nn_descriptors_model( **nn_descriptors_model_inputs )
    nn_training_each_solvent = nn_model.train_across_data( column_name = 'cosolvent')
    
    ## CREATING DICTIONARY FOR EACH
    cosolvents = ['DIO', 'GVL', 'THF', 'DIO_GVL_THF']
    metric_types = ['slope', 'rmse']
    
    ## CREATING DICTIONARY
    nn_dict = {'NN_model': {}}
    
    ## LOOPING THROUGH EACH ONE
    for each_cosolvent in cosolvents:
        for each_metric in metric_types:
            ## DEFINING NAME
            name = each_cosolvent + '_' + each_metric
            
            ## IF ALL COSOLVENTS
            if each_cosolvent == 'DIO_GVL_THF':
                value = nn_model.predict_stats[0][each_metric]
            else:
                value = nn_training_each_solvent[each_cosolvent]['predict_stats'][0][each_metric]
                
            ## STORING INTO DICTIONARY
            nn_dict['NN_model'][name] = value
                
    ## CREATING DATAFRAME
    nn_df = pd.DataFrame(nn_dict)
    nn_df = nn_df.transpose()
    
    ## STORING
    df = df.append(nn_df)
    
    ## OUTPUTTING
    df.to_csv(path_output_cross_csv)
    print("Writing to %s"%(path_output_cross_csv) )
    
    #%%
    
    ###########################################################################
    ### TABLE S3: CROSS VALIDATION ACROSS REACTANT AND COSOLVENTS
    ###########################################################################
    
    ## DEFINING PATH DICT TO CROSS VALIDATION PATHS
    path_to_cross_validation = simulation_path_dicts['cross_validation_paths']

    ## DEFINING RESULTS PICKLE
    results_pickle_file = r"model.results" # DIO 
        
    ## DEFINING CROSS VALIDATION INPUTS
    cross_valid_inputs = {
            'combined_database_path': combined_database_path,
            'class_file_path': class_file_path,
            'image_file_path': path_dict['path_image_dir'],
            'sim_path': sim_path,
            'database_path': database_path,
            'results_pickle_file': results_pickle_file,
            'verbose': True,
            }
    
    ## GENERATING CROSS VALIDATION CLASS
    cross_valid = extract_cross_validation()
    
    ## GETTING STORAGE    
    cross_validation_storage = cross_valid.load_multiple_cross_validations(path_to_cross_validation = path_to_cross_validation,
                                                                           pickle_path = path_dict['path_pickle'],
                                                                           cross_valid_inputs = cross_valid_inputs,
                                                                           )
    
    ## CROSS VALIDATION SLOPE AND RMSE
    df_storage_crossvalid = cross_valid.get_df_multiple_cross_validations(cross_validation_storage = cross_validation_storage,
                                                                          desired_stats = ['slope', 'rmse', 'pearson_r'],
                                                                          output_csv_name = "SI_TABLE3_CROSS_VALIDATION",
                                                                          path_output_excel_spreadsheet = path_output_excel_spreadsheet,
                                                                          )
    
    ## CROSS VALIDATION TEST SET DATA
    df_storage = cross_valid.get_df_test_set_stats(cross_validation_storage = cross_validation_storage,
                                                      desired_stats = ['slope', 'rmse', 'pearson_r'],
                                                      path_output_excel_spreadsheet = path_output_excel_spreadsheet,
                                                      output_csv_name = 'SI_TABLE3_STATS'                                      
                                                      )
        

    #%%
    ###########################################################################
    ### SI FIGURE S4 -- FIGURE FOR MULTIPLE VOXEL REPRESENTATIONS
    ###########################################################################
    ## DEFINING PICKLE NAMES
    INSTANCES_DICT = {
            'main_text': r'20_20_20_20ns_oxy_3chan-split_avg_nonorm-10-CEL_ETBE_FRU_LGA_PDO_XYL_tBuOH-DIO_GVL_THF-10_25_50_75',
            '3chan_all_atoms': r'20_20_20_20ns_firstwith10-split_avg_nonorm-10-CEL_ETBE_FRU_LGA_PDO_XYL_tBuOH-DIO_GVL_THF-10_25_50_75',
            '3chan_hydroxy': r'20_20_20_20ns_3channel_hydroxyl_firstwith10-split_avg_nonorm-10-CEL_ETBE_FRU_LGA_PDO_XYL_tBuOH-DIO_GVL_THF-10_25_50_75',
            '4chan_oxy': r'20_20_20_20ns_firstwith10_oxy-split_avg_nonorm-10-CEL_ETBE_FRU_LGA_PDO_XYL_tBuOH-DIO_GVL_THF-10_25_50_75',
            '2chan_solvent': r'20_20_20_20ns_solvent_only_firstwith10-split_avg_nonorm-10-CEL_ETBE_FRU_LGA_PDO_XYL_tBuOH-DIO_GVL_THF-10_25_50_75',
            }
    
    ## DEFINING FIGURE SIZE
    figsize = [8, 6]
    
    ## DEFINING INSTANCE
    specific_instance = "XYL_403.15_DIO_10"
    specific_partition = 0
    
    #### MAIN TEXT
    ## LOOPING
    for idx, instance_key in enumerate(INSTANCES_DICT):
        #    ## DEFINING INSTANCE KEY
        #    instance_key = 'main_text'
    
        ## LOADING INSTANCES
        instances = load_instances_based_on_pickle(pickle_name = INSTANCES_DICT[instance_key] ,
                                                   path_combined_database = path_dict['combined_database_path'])
        
        ## PLOTTING
        fig, ax = publish_voxel_image(instances = instances, 
                            instance_name = specific_instance,
                            specific_partition = specific_partition,
                            plot_voxel_split_inputs = {
                                    'alpha': 0.3,
                                    'want_renormalize': True,
                                    # 'tick_limits': np.arange(0, 32+4, 8), # np.append(, 31), # np.append(np.arange(0, 31, 4), 31), # np.arange(0, 33, 4) , 
                                    'figsize': figsize,
                                    'want_separate_axis': True
                                    }
                            )
        
        ## TURNING OFF LABELS
        # if idx != 0:
        [turn_ax_labels_off(each_axis) for each_axis in ax]
        
        ## STORING FIGURE
        for idx, each_fig in enumerate(fig):
            store_figure( fig = each_fig, 
                          path = os.path.join(path_image_dir, instance_key + '_' + str(idx) ),
                          save_fig = save_fig,
                          fig_extension = fig_extension,
                          dpi = 300, # Lower dpi for storing such a large picture
                         )

    
    
    #%%
    
    ###########################################################################
    ### SI FIGURE S5: 16 x 16 x 16, 20 x 20 x 20, and 32 x 32 x 32 REPS
    ###########################################################################
    
    ## DEFINING FIGURE SIZE
    figsize = [8, 6]
    
    ## DEFINING INSTANCE
    specific_instance = "XYL_403.15_DIO_10"
    specific_partition = 0
    
    ## DEFINING PICKLE NAMES
    INSTANCES_DICT = {
            '16_16_16': r'16_16_16_20ns_oxy_3chan_firstwith10-split_avg_nonorm-10-CEL_ETBE_FRU_LGA_PDO_XYL_tBuOH-DIO_GVL_THF-10_25_50_75',
            '20_20_20': r'20_20_20_20ns_oxy_3chan-split_avg_nonorm-10-CEL_ETBE_FRU_LGA_PDO_XYL_tBuOH-DIO_GVL_THF-10_25_50_75',
            '32_32_32': r'32_32_32_20ns_oxy_3chan_firstwith10-split_avg_nonorm-10-CEL_ETBE_FRU_LGA_PDO_XYL_tBuOH-DIO_GVL_THF-10_25_50_75',
            }
    
    ## DEFINING TICK RANGE
    TICK_RANGES_SIZES={
            '16_16_16': np.arange(0, 16+4, 4),
            '20_20_20': None,
            '32_32_32': np.arange(0, 32+4, 8),
            }
    
    #### MAIN TEXT
    ## LOOPING
    for idx, instance_key in enumerate(INSTANCES_DICT):
        ## LOADING INSTANCES
        instances = load_instances_based_on_pickle(pickle_name = INSTANCES_DICT[instance_key] ,
                                                   path_combined_database = path_dict['combined_database_path'])
        
        ## PLOTTING
        fig, ax = publish_voxel_image(instances = instances, 
                            instance_name = specific_instance,
                            specific_partition = specific_partition,
                            plot_voxel_split_inputs = {
                                    'alpha': 0.3,
                                    'want_renormalize': True,
                                    'tick_limits': TICK_RANGES_SIZES[instance_key],
                                    'figsize': figsize,
                                    'want_separate_axis': False
                                    }
                            )

        
        ## STORING FIGURE
        store_figure( fig = fig, 
                      path = os.path.join(path_image_dir, instance_key ),
                      save_fig = save_fig,
                      fig_extension = fig_extension,
                      dpi = 300, # Lower dpi for storing such a large picture
                     )

    #%%
    
    ###########################################################################
    ### TABLE S4: SLOPE AND RMSE FOR EACH SOLVENT SYSTEM
    ###########################################################################
    
    ## DEFINING RESULTS PICKLE
    results_pickle_file = r"model.results" # DIO 
    
    ## DEFINING LIST OF DIRECTORIES TO LOOK IN (3D CNNS)
    list_of_directories = [
            r'190731-parity_20_20_20_20ns_solvent_only_firstwith10', ## 2 channel 
            r'190725-newrep_20_20_20_20ns_oxy_3chan', ## 3 channel, oxygen atom reactants <-- may have
            r'190731-parity_20_20_20_20ns_3channel_hydroxyl_firstwith10', ## 3 channel, hydroxyl reactants
            r'190705-20_20_20_20ns_firstwith10_ALL_SOLVENTS', ## 3 channel, all atom reactants
            r'190718-oxyrep_20_20_20_20ns_firstwith10_oxy', # 4 channel oxy
            r'190805-different_size_16_16_16_20ns_oxy_3chan_firstwith10', ## 16 x 16 x 16
            r'190730-parity_32_32_32_20ns_oxy_3chan_firstwith10', ## 32 x 32 x 32
            r'190802-parity_vgg16_32_32_32_20ns_oxy_3chan_firstwith10', # VGG16
            ]
    
    ## CREATING DATAFRAME
    df = generate_dataframe_slope_rmse(sim_path = path_dict['sim_path'],
                                       list_of_directories = list_of_directories,
                                       results_pickle_file = 'model.results',
                                       desired_row_label = None,
                                       desired_network = 'solvent_net',
                                       )
    ## DEFINING OUTPUT CSV FILE
    path_output_cross_csv = os.path.join(path_output_excel_spreadsheet, "SI_TABLE4_SLOPE_RMSE.csv")
    
    ## OUTPUTTING
    df.to_csv(path_output_cross_csv)
    print("Writing to %s"%(path_output_cross_csv) )
    
    #%%
    
    ############ SCRIPT TO GET CROSS VALIDATION DATAFRAMES ######################
    ###########################################################################
    ### TABLE S4: Cross validation across different voxel representation inputs
    ############################################################################

    ## DEFINING NEW PATHS
    cross_rep_paths = {

#            ## SOLVENT ONLY
#            r"2_chan_no_react_Solute": r"190731-diffrep_cross_val-20_20_20_20ns_solvent_only_firstwith10-solvent_net-solute",
#            r"2_chan_no_react_Cosolvent": r"190731-diffrep_cross_val-20_20_20_20ns_solvent_only_firstwith10-solvent_net-cosolvent",
#            
#            ## 3 CHAN OXY
#            r"3_chan_oxy_react_Solute" : r"190725-cross_valid-newrep-20_20_20_20ns_oxy_3chan-solvent_net-solute",
#            r"3_chan_oxy_react_Cosolvent" : r"190725-cross_valid-newrep-20_20_20_20ns_oxy_3chan-solvent_net-cosolvent",
#            ## 3 CHAN HYDROXY
#            r"3_chan_hydroxy_react_Solute": r"190731-diffrep_cross_val-20_20_20_20ns_3channel_hydroxyl_firstwith10-solvent_net-solute",
#            r"3_chan_hydroxy_react_Cosolvent": r"190731-diffrep_cross_val-20_20_20_20ns_3channel_hydroxyl_firstwith10-solvent_net-cosolvent",
#            
#            ## ALL REACTANT ATOM
#            'SolventNet_Solute': r'190705-20_20_20_20ns_firstwith10_solute-solvent_net',
#            'SolventNet_Cosolvent': r'190705-20_20_20_20ns_firstwith10_cosolvent-solvent_net' ,
#            
#            ## 4 CHAN HYDROXY
#            '4_chan_oxy_Solute': r"190726-cross_valid-rerun-20_20_20_20ns_firstwith10_oxy-solvent_net-solute" ,
#            '4_chan_oxy_Cosolvent': r"190726-cross_valid-rerun-20_20_20_20ns_firstwith10_oxy-solvent_net-cosolvent" ,
#
#            ## ADDING 16 X 16 X 16 
#            r"16_16_16_3chan_oxy_Solute" : r"190805-cross_val_size-16_16_16_20ns_oxy_3chan_firstwith10-solvent_net-solute",
#            r"16_16_16_3chan_oxy_Cosolvent" : r"190805-cross_val_size-16_16_16_20ns_oxy_3chan_firstwith10-solvent_net-cosolvent",
#            
#            ## 32 x 32 x 32
#            r"32_32_32_3chan_oxy_Solute" : r"190730-3chan_cross_val-32_32_32_20ns_oxy_3chan_firstwith10-solvent_net-solute",
#            r"32_32_32_3chan_oxy_Cosolvent" : r"190730-3chan_cross_val-32_32_32_20ns_oxy_3chan_firstwith10-solvent_net-cosolvent",
            
            ## VGG16
            r"VGG16_Solute" : r"190802-cross_val_planar-32_32_32_20ns_oxy_3chan_firstwith10-vgg16-solute",
            r"VGG16_Cosolvent": r"190802-cross_val_planar-32_32_32_20ns_oxy_3chan_firstwith10-vgg16-cosolvent",
            
            }
    
    ## DEFINING PATH
    path_to_cross_validation = cross_rep_paths
    
    ## DEFINING CROSS VALIDATION INPUTS
    cross_valid_inputs = {
            'combined_database_path': combined_database_path,
            'class_file_path': class_file_path,
            'image_file_path': path_dict['path_image_dir'],
            'sim_path': sim_path,
            'database_path': database_path,
            'results_pickle_file': results_pickle_file,
            'verbose': True,
            }
    
    ## GENERATING CROSS VALIDATION CLASS
    cross_valid = extract_cross_validation()
    
    ## GETTING STORAGE    
    cross_validation_storage = cross_valid.load_multiple_cross_validations(path_to_cross_validation = path_to_cross_validation,
                                                                           pickle_path = path_dict['path_pickle'],
                                                                           cross_valid_inputs = cross_valid_inputs,
                                                                           )
    
    ## CROSS VALIDATION SLOPE AND RMSE
    df_storage_crossvalid = cross_valid.get_df_multiple_cross_validations(cross_validation_storage = cross_validation_storage,
                                                                          desired_stats = ['rmse'],
                                                                          output_csv_name = "SI_TABLE4_CROSS_VALIDATION",
                                                                          path_output_excel_spreadsheet = path_output_excel_spreadsheet,
                                                                          )
    
    ## CROSS VALIDATION TEST SET DATA
    df_storage = cross_valid.get_df_test_set_stats(cross_validation_storage = cross_validation_storage,
                                                      desired_stats = ['slope', 'rmse', 'pearson_r'],
                                                      path_output_excel_spreadsheet = path_output_excel_spreadsheet,
                                                      output_csv_name = 'SI_TABLE4_STATS'                                      
                                                      )


    #%%
    ############ SCRIPT TO GET PREDICTION ACCURACY AND PLOTS ############
    ############################################################################
    ### TABLE S4, PREDICTION ACCURACY
    ############################################################################
    ## DEBUGGING FOR NEW SIMS
    ## FINDING SIM PATH
    sim_path = path_dict['sim_path']    

    ## DEFINING NEW LIST
    simulation_path_dicts_new_rep = {
#            '2_chan_no_react': os.path.join(r"190731-parity_20_20_20_20ns_solvent_only_firstwith10",
#                                            r"20_20_20_20ns_solvent_only_firstwith10-split_avg_nonorm-10-strlearn-0.80-solvent_net-500-CEL_ETBE_FRU_LGA_PDO_XYL_tBuOH-10_25_50_75-DIO_GVL_THF"),
#            '3_chan_oxy': os.path.join(r"190725-newrep_20_20_20_20ns_oxy_3chan",
#                                       r"20_20_20_20ns_oxy_3chan-split_avg_nonorm-10-strlearn-0.80-solvent_net-500-CEL_ETBE_FRU_LGA_PDO_XYL_tBuOH-10_25_50_75-DIO_GVL_THF"),
#    
#            '3_chan_hydroxy': os.path.join(r"190731-parity_20_20_20_20ns_3channel_hydroxyl_firstwith10",
#                                           r"20_20_20_20ns_3channel_hydroxyl_firstwith10-split_avg_nonorm-10-strlearn-0.80-solvent_net-500-CEL_ETBE_FRU_LGA_PDO_XYL_tBuOH-10_25_50_75-DIO_GVL_THF"),
#    
#            '4_chan_oxy': os.path.join( r"190718-oxyrep_20_20_20_20ns_firstwith10_oxy",
#                                        r"20_20_20_20ns_firstwith10_oxy-split_avg_nonorm-10-strlearn-0.80-solvent_net-500-CEL_ETBE_FRU_LGA_PDO_XYL_tBuOH-10_25_50_75-DIO_GVL_THF"),
#            '3_chan_all_solvents': os.path.join( r"190705-20_20_20_20ns_firstwith10_ALL_SOLVENTS",
#                                        r"20_20_20_20ns_firstwith10-split_avg_nonorm-10-strlearn-0.80-solvent_net-500-CEL_ETBE_FRU_LGA_PDO_XYL_tBuOH-10_25_50_75-DIO_GVL_THF"),
#
#            '16_16_16_3chan_oxy': os.path.join(r"190805-different_size_16_16_16_20ns_oxy_3chan_firstwith10",
#                                            r"16_16_16_20ns_oxy_3chan_firstwith10-split_avg_nonorm-10-strlearn-0.80-solvent_net-500-CEL_ETBE_FRU_LGA_PDO_XYL_tBuOH-10_25_50_75-DIO_GVL_THF"),
#            '32_32_32_3chan_oxy': os.path.join(r"190730-parity_32_32_32_20ns_oxy_3chan_firstwith10",
#                                            r"32_32_32_20ns_oxy_3chan_firstwith10-split_avg_nonorm-10-strlearn-0.80-solvent_net-500-CEL_ETBE_FRU_LGA_PDO_XYL_tBuOH-10_25_50_75-DIO_GVL_THF"),
            ## VGG16
            'vgg16': simulation_path_dicts['3D_CNN_Training_All_Solvents_vgg16'],
            }
    
    ## DEFINING LIST
    parity_plot_list = simulation_path_dicts_new_rep.keys()
    
    ## ADDING TO KEYS
    simulation_paths_full_new_rep = { each_key: os.path.join( path_dict['sim_path'], simulation_path_dicts_new_rep[each_key] ) for each_key in simulation_path_dicts_new_rep.keys()}
    
    ## DEFINING MODEL WEIGHTS
    model_weights = "model.hdf5"
    path_pickle = path_dict['path_pickle']
    want_repickle = False
    num_partitions = 2
    ## CREATING DICTIONARY
    predict_storage_dict = {}
    
    ## DEFINING INPUTS FOR LOOP GRID
    loop_database_and_pred_inputs={
            'path_pickle': path_pickle,
            'want_repickle': want_repickle,
            'num_partitions': num_partitions,
            }
    
    ## DEFINING PREDICTIVE MODEL
    predict_model = publish_predict_model()
    
    ## LOOPING THROUGH DATABASE
    predict_storage_dict = predict_model.loop_multiple_simulation_paths(simulation_path_dict = simulation_paths_full_new_rep,
                                                                        loop_database_and_pred_inputs = loop_database_and_pred_inputs,
                                                                        model_weights = model_weights)
    
    ## DEFINING INPUTS
    csv_input_dict = {
            'predict_storage_dict': predict_storage_dict,
            'csv_file_name': 'SI_TABLE_S4_PREDICTIONS',
            'path_output_excel_spreadsheet': path_output_excel_spreadsheet,
            'desired_metrics': ['rmse'],
            }
    
    ## GENERATING CSV STATS
    predict_model.generate_csv_stats_multiple(**csv_input_dict)
    
    #%%
    
    ### TAKING 32 X 32 X 32 CASE FROM PREVIOUS FIGURE S5
    ############################################################################
    ### SI FIGURE S6 -- PLANAR CASES  -- VGG16
    ############################################################################
    ## DEFINING PICKLE NAME
    pickle_name = r"32_32_32_20ns_oxy_3chan_firstwith10-split_avg_nonorm_planar-10-CEL_ETBE_FRU_LGA_PDO_XYL_tBuOH-DIO_GVL_THF-10_25_50_75"
    
    ## LOADING INSTANCES
    instances = load_instances_based_on_pickle(pickle_name = pickle_name ,
                                               path_combined_database = path_dict['combined_database_path'])

    ## DEFINING DEFAULT FIGURE NAME
    fig_name = "SI_6A_2D_planar"
    ## DEFINING FIGURE SIZE
    figsize = [6, 6]
    ## DEFINING SPECIFIC INDEX
    specific_indexes = [0, 1, 2] # x, y, z
    
    
    ## DEFINING INSTANCE
    specific_instance = "XYL_403.15_DIO_10"
    specific_partition = 0
    
    ## DEFINING FIGURE NAME
    fig_name =  fig_name + "_" + specific_instance
    
    ## FINDING INSTANCE
    index_instance = instances.instance_names.index( specific_instance )
    
    ## LOOPING THROUGH EACH INDEX
    for each_index in specific_indexes:
        ## ADDING TO FIGURE NAME
        current_fig_name = fig_name + '_' + str(each_index)
        
        ## PLOTTING
        fig, ax = publish_voxel_image(  instances = instances, 
                                        instance_name = specific_instance,
                                        specific_partition = each_index,
                                        plot_voxel_split_inputs = {
                                                'alpha': 0.3,
                                                'want_renormalize': True,
                                                'tick_limits': np.arange(0, 32+8, 8), # np.append(, 31), # np.append(np.arange(0, 31, 4), 31), # np.arange(0, 33, 4) , 
                                                'ax_limits': np.array([-0.5,31.5]),
                                                'figsize': figsize,
                                                }
                                        )
        
        ## STORING FIGURE
        store_figure( fig = fig, 
                      path = os.path.join(path_image_dir, current_fig_name),
                      save_fig = save_fig,
                      fig_extension = 'eps', # fig_extension, <-- eps for some reason works and svg does not
                      dpi = 300, # Lower dpi for storing such a large picture
                     )
        

    #%%
    ############################################################################
    ### SI FIGURE 6B: Parity plot for VGG16
    ############################################################################
    ## DEFINING FIGNAME
    figname = "SI_6B_VGG16_"
    
    ## DEFINING RESULTS PICKLE
    results_pickle_file = r"model.results" # THF
    
    ## DEFINING NETWORKS
    networks = [ "vgg16" ]
    
    ## DEFINING FIGURE SIZE
    figure_size= FIGURE_SIZES_CM_SI['2_col']/3
        
    ## LOOPING THROUGH EACH NETWORK
    for each_network in networks:
        current_figname = figname + each_network
        ## DEFINING VARIABLES
        results_file_path =simulation_path_dicts['3D_CNN_Training_All_Solvents_' + each_network]
        ## DEFINING FULL PATH
        results_full_path = os.path.join( results_file_path, results_pickle_file )
        ## DEFINING ANALYSIS
        analysis = pd.read_pickle( results_full_path  )[0]
        
        df = analysis.dataframe # dataframe.set_index('solute')
        
        ## PLOTTING PARITY
        plot_parity_publication_single_solvent_system( dataframe = df,
                                                      fig_name = os.path.join(path_image_dir, current_figname) + '.' + fig_extension,
                                                       mass_frac_water_label = 'mass_frac',
                                                       sigma_act_label = 'y_true',
                                                       sigma_pred_label = 'y_pred',
                                                       sigma_pred_err_label = 'y_pred_std',
                                                       fig_extension = fig_extension, # fig_extension
                                                       save_fig_size = figure_size, # (16.8/3, 16.8/3),
                                                       save_fig = save_fig)
    #%%
    ############################################################################
    ### SI FIGURE 6C: Learning curve for VGG16
    ############################################################################
    
    ## DEFINING FIGURE NAME
    fig_name = "SI_6C_VGG16_learning_curve"
    
    ## DEFINING FIGURE SIZE
    figure_size= FIGURE_SIZES_CM_SI['2_col']/3
    
    ## CREATING IMAGE
    learning_curve_vgg16 = publish_plot_learning_curve(figure_size = figure_size)
    
    ## DEFINING PICKLE
    results_pickle = "model.results"
    
    ## DEFINING VGG16 PATH
    path_to_results = os.path.join( simulation_path_dicts['3D_CNN_Training_All_Solvents_vgg16'], results_pickle )
    
    ## LOADING
    analysis = pd.read_pickle( path_to_results )[0]
    
    ## PLOTTING LEARNING CURVE
    learning_curve_vgg16.plot_each_loss_per_color( history = analysis.history,
                                             network_type = 'vgg16')
    ## FINALIZING LEARNING CURVE
    learning_curve_vgg16.finalize_image()
    
    ## SETTING X LIM
    learning_curve_vgg16.ax.set_xticks(np.arange(0,500 + 100, 100))
    
    ## STORING FIGURE
    store_figure( fig = learning_curve_vgg16.fig, 
                  path = os.path.join(path_image_dir, fig_name),
                  save_fig = save_fig,
                  fig_extension = fig_extension,
                 )
    #%%
    ############################################################################
    ### SI FIGURE 6DE: Cross validation across cosolvents and reactants
    ############################################################################
    
    ## DEFINING FIGURE SIZE
    figure_size=( 18.542/3, 18.542/3 )
    
    ## DEFINING MAIN DIRECTORY LIST
    main_dir_dict={
            'reactant': os.path.basename( SI_cross_validation_path['vgg16_Solute'] ) ,
            'cosolvent': os.path.basename( SI_cross_validation_path['vgg16_Cosolvent'] ),
            }
    ## STORING DFS
    df_storage = []
    
    ## DEFINING FIGURE NAME
    fig_name = "SI_6DE_cross_valid"
    
    ## CREATING CROSS VALIDATION
    cross_valid_extracted = extract_cross_validation()
    
    ## DEFINING MAIN DIRECTORY
    for main_dir_key in main_dir_dict:
        ## NEW FIGURE NAME
        current_fig_name = fig_name + '_' + main_dir_key
        ## DEFINING MAIN DIRECTORY
        main_dir = main_dir_dict[main_dir_key]
        
        ## DEFINING INPUTS
        cross_valid_inputs = {
                'main_dir': main_dir,
                'combined_database_path': combined_database_path,
                'class_file_path': class_file_path,
                'image_file_path': path_dict['path_image_dir'],
                'sim_path': sim_path,
                'database_path': database_path,
                'results_pickle_file': results_pickle_file,
                'verbose': True,
                }
        
        ## LOADING CROSS VALIDATION
        cross_valid_results = cross_valid_extracted.load_cross_validation(cross_valid_inputs  = cross_valid_inputs,
                                                                          pickle_path = path_dict['path_pickle'])
        
        ## DEFINING PLOTTING INPUTS
        parity_plot_inputs = \
            {
                    'save_fig_size': figure_size,
                    'save_fig': save_fig,
                    'fig_name': os.path.join(path_image_dir, current_fig_name) + '.' + fig_extension,
                    'fig_extension': fig_extension,
                    }
        ## PLOTTING CROSS VALIDATION
        fig, ax = cross_valid_extracted.plot_parity_plot(cross_valid_results = cross_valid_results,
                                                         parity_plot_inputs = parity_plot_inputs,
                                                         want_combined_plot = True)
        
    #%%
    ############################################################################
    ### SI FIGURE 6FGH: Predictive model
    ############################################################################
    
    ## DEFINING PATH TO SIMULATION
    path_to_sim =simulation_path_dicts['3D_CNN_Training_All_Solvents_vgg16']    
    
    ## DEFINING SIM NAME
    sim_name = os.path.basename(path_to_sim)
    
    ## DEFINING MAIN DIRECTORY
    main_dir = os.path.basename(os.path.dirname(path_to_sim))
    
    ## DEFINING MODEL WEIGHTS
    model_weights = "model.hdf5"
    
    ## DEFINING FULL PATH TO MODEL
    path_model = os.path.join(path_to_sim, model_weights)
    
    ## DEFINING INPUTS FOR PREDICTED MODEL
    inputs_predicted_model = {
            'path_model': path_model,
            'verbose': True,
            }
    ## LOADING MODEL
    trained_model = predict_with_trained_model( **inputs_predicted_model )
    
    ## DEFINING PREDICTIVE MODEL
    predict_model = publish_predict_model()
    
    ## DEFINING DATABASE
    test_database_basename = sim_name.split('-')[0] + '_' #  '20_20_20_20ns_oxy_3chan_'
    
    ## GETTING DATABASE DICT
    database_dict = get_test_pred_test_database_dict(test_database_basename = test_database_basename)
    
    ## LOOPING AND PREDICTING
    stored_predicted_value_list_storage, figure_name_list = predict_model.loop_database_and_predict(trained_model = trained_model,
                                                                                                    main_dir = main_dir,
                                                                                                    path_pickle = path_dict['path_pickle'],
                                                                                                    database_dict = database_dict,
                                                                                                    want_repickle = False,
                                                                                                    num_partitions = 2,
                                                                                                    )
    
    ## DEFINING FIGURE SIZE
    figure_size=FIGURE_SIZES_CM_SI['2_col']/3
    # ( 18.542/3, 18.542/3 )
    
    ## DEFINING PARITY PLOT INPUTS    
    parity_plot_inputs = \
        {
                'save_fig_size': figure_size,
                'save_fig': save_fig,
                'fig_extension': fig_extension,
                }       
        
    predict_model.plot_parity_plot(stored_predicted_value_list_storage = stored_predicted_value_list_storage,
                                   figure_name_list = figure_name_list,
                                   parity_plot_inputs = parity_plot_inputs,
                                   output_path = path_dict['path_image_dir'],
                                   )
    