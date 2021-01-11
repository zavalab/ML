#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plotting_scripts.py
This contains plotting scripts
VARIABLES:
    DEFAULT_FIG_INFO: default figure size information (deprecitated)
    MASS_FRAC_SYMBOLS_DICT: dictionary for symbols
    COSOLVENT_FILL_DICT: dictionary for c osolvent
    SOLUTE_COLOR_DICT: dictionary for solute color
    SOLUTE_ORDER: dictionary for solute order

FUNCTIONS:
    rename_df_column_entries: renames dataframe column entries
    order_df: functions that order df based on a column name
    cm2inch: function that converts cm to inches
    create_fig_based_on_cm: function that creates figure  based on input cms
    renormalize_rgb_array: code to renormalize rgb array based on the channel
    plot_voxel: function that plots the voxels
    update_ax_limits: function that updates axis limits
    change_axis_label_fonts: function that changes axis label fonts
    get_cmap: function that gets cmap -- generates a color bar of distinct colors

"""
## IMPORTING MODULES
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import numpy as np
import pandas as pd

## EDITING DEFAULTS
import matplotlib as mpl
mpl.rcParams['axes.linewidth'] = 1.0
mpl.rcParams['xtick.major.width'] = 1.0
mpl.rcParams['ytick.major.width'] = 1.0

    
PLOT_INFO = {
        'markersize': 6,
        }

DEFAULT_FIG_INFO={
        # 'figsize'   : (4,6), # Width, height
        'figsize'   : (3.54331,3.54331),
        'dpi'       : 300,
        'facecolor' : 'w',
        'edgecolor' : 'k',
        }
LABELS_DICT={
        'fontname': 'Arial',
        'fontsize': 10,
        }

TICKS_DICT={
        'fontname': 'Arial',
        'fontsize': 8,
        }

        
AXIS_LABELS_DICT={
        # 'fontname': 'Arial',
        'fontsize': 8,
        }

LINE_STYLE={
        "linewidth": 1.6, # width of lines
        }

AXIS_RANGES={
        'hyperparameter':
            {'x': (0, 20.5, 1), # min, max, inc.
             'y': (0.96, 0.98, 0.005 )  },
        'hyperparameter_lower_bound':
            {'x': (0, 20.5, 1), # min, max, inc.
             'y': (0.90, 0.98, 0.005 )  },             
        'roc_curve':
            {'x': (0, 1.01, 0.1), # min, max, inc.
             'y': (0, 1.01, 0.1 )  },
        'precision_recall':
            {'x': (0, 1.01, 0.1), # min, max, inc.
             'y': (0, 1.01, 0.1 )  },
        'F1_train_test_per_epoch':
            {'x': (0, 31, 5), # min, max, inc.
             # 'y': (0, 1.01, 0.1 )  },
             # 'y': (0.5, 1.01, 0.1 )  },
             'y': (0.5, 0.8, 0.1 )  },
        'test_accuracy_vs_tree':
            {'x': (0.0, 8.1, 1), # min, max, inc.
             'y': (0.0, 1.05, 0.1 )  },
        'learning_curve':
            {'x': None, # min, max, inc.
             'y': (0.0, 1.05, 0.2 )  },
        }
SAVE_FIG_INFO={
        # 'bbox_inches': 'tight',
        'dpi'        : 600,
        }

## DEFINING FONT SIZE
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

## STORING FONT INFORMATION
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

## DEFINING DICTIONARIES
# Other symbols: https://matplotlib.org/3.1.0/gallery/lines_bars_and_markers/marker_reference.html
MASS_FRAC_SYMBOLS_DICT={
        '0.10': "D", # Diamond -- 90 wt% org
        '0.12': "*", # Star -- 90 wt% org
        '0.25': "v", # Lower triangular -- 75 wt% org
        '0.35': ">", # Lower triangular -- 75 wt% org
        '0.50': "^", # Upper triangular -- 50 wt% org
        '0.56': "h", # Upper triangular -- 50 wt% org
        '0.75': "o", # Circular -- 25 wt% org
        }

## DEFINING COSOLVENT FILL SHAPE
# https://matplotlib.org/gallery/lines_bars_and_markers/marker_fillstyle_reference.html
COSOLVENT_FILL_DICT={
        'DIO': 'left',
        'GVL': 'full', 
        'THF': 'none',
        'dmso': 'bottom',
        'ACE': 'right',
        'ACN': 'top',
        }

## DEFINING SOLUTE COLORS
#SOLUTE_COLOR_DICT={
#        'ETBE'  : 'blue',
#        'TBA'   : 'black',
#        'tBuOH' : 'black', # replica of TBA
#        'LGA'   : 'brown',
#        'PDO'   : 'cyan',
#        'FRU'   : 'green',
#        'CEL'   : 'pink',
#        'XYL'   : 'red',
#        'GLU'   : 'purple',
#        }

SOLUTE_COLOR_DICT={
        'ETBE'  : 'blue',
        'TBA'   : 'black',
        'tBuOH' : 'black', # replica of TBA
        'LGA'   : 'brown',
        'PDO'   : 'cyan',
        # 'FRU'   : 'green',
        'FRU'   : 'lightgreen',
        'CEL'   : 'purple',
        'XYL'   : 'red',
        'GLU'   : 'pink',
        }

## DEFINING SOLVENT ORDER
SOLUTE_ORDER = [ 'ETBE', 
                  'TBA',
                  'LGA',
                  'PDO',
                  'FRU',
                  'GLU',
                  'CEL',
                  'XYL',
                  ]
### FUNCTION TO CONVERT TIME
def convert_hms_to_Seconds(hms):
    h = hms[0]
    m = hms[1]
    s = hms[2]
    total_time_seconds = h * 60 * 60 + m * 60 + s
    return total_time_seconds

### FUNCTION RENAME DF
def rename_df_column_entries(df,
                             col_name = 'solute',
                             change_col_list = [ 'tBuOH', 'TBA' ]
                             ):
    '''
    The purpose of this function is to rename df column entries.
    INPUTS:
        df: [pd.dataframe]
            pandas dataframe
        col_name: [str]
            column name
        change_col_list: [list]
            list of columns we want to change
    OUTPUTS:
        updated df (changed in place)
    '''
    ## CHANGING COLUMN NAMES (IF NECESSARY)
    df.loc[df.solute == change_col_list[0], col_name] = change_col_list[-1]
    return df

### FUNCTION TO ORDER DF
def order_df(df,
             ordered_classes = SOLUTE_ORDER,
             col_name = 'solute',
             ):
    '''
    This function orders a dataframe based on an input list
    INPUTS:
        df: [pd.dataframe]
            pandas dataframe
        col_name: [str]
            column name
        ordered_classes: [list]
            ordered classes
    OUTPUTS:
        ordered_df: [pd.dataframe]
            ordered pandas dataframe based on your input list. Note that 
            this code only outputs the information given as a list
    '''
    
    ## CREATING EMPTY DF LIST
    df_list = []

    for i in ordered_classes:
       df_list.append(df[df['solute']==i])
    
    ordered_df = pd.concat(df_list)
    return ordered_df

## FUNCTION TO CONVERT FIGURE SIZE
def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)
    
## FUNCTION TO CREATE FIG BASED ON CM
def create_fig_based_on_cm(fig_size_cm = (16.8, 16.8)):
    ''' 
    The purpose of this function is to generate a figure based on centimeters 
    INPUTS:
        fig_size_cm: [tuple]
            figure size in centimeters 
    OUTPUTS:
        fig, ax: 
            figure and axis
    '''
    ## FINDING FIGURE SIZE
    figsize=cm2inch( *fig_size_cm )
    ## CREATING FIGURE
    fig = plt.figure(figsize = figsize) 
    ax = fig.add_subplot(111)
    return fig, ax

### FUNCTION TO RENORMALIZE THE DATA
def renormalize_rgb_array( rgb_array ):
    '''
    The purpose of this function is to renormalize any RGB array. The shape of 
    the input array is 1 x 20 x 20 x 20 x 3. The output array is the same shape, but 
    we are normalizing each of the final three dimensions (R, G, B).
    INPUTS:
        rgb_array: [array, shape=(1,N,N,N,3)]
            rgb array in volumetric form. The "1" is for each frame.
    OUTPUTS:
        renormalized_rgb_array: [array, shape=(1,N,N,N,3)]
            rgb array such that R, G, B ranges from 0 to 1. Normalization is 
            important for feature inputs. You can test that max rgb array is 1 by:
            np.max( updated_rgb_array[...,0] )
    '''
    ## CREATING COPY OF ARRAY
    renormalized_rgb_array = np.copy(rgb_array)
    ## LOOPING THROUGH EACH DIMENSION
    for each_dim in range(renormalized_rgb_array.shape[-1]):
        renormalized_rgb_array[..., each_dim] /= np.max( renormalized_rgb_array[...,each_dim] )
    return renormalized_rgb_array

### FUNCTION TO PLOT VOXEL
def plot_voxel(grid_rgb_data, 
               frame = 0, 
               want_split=False,
               want_renormalize = False, 
               verbose = False):
    '''
    This functions plots the voxel:
        red: water
        blue: cosolvent
        green: reactant
    IMPORTANT NOTES: 
        - Check if your voxels are normalized between 0 to 1 in terms of RGB format. 
        - Otherwise, you will get voxels that do not make sense (i.e. black box)
        - This code worked for python 3.5
    INPUTS:
        self:
            class object
        frame: [int]
            frame you are interested in plotting
            if frame = None, we will assume that rgb data is not time dependent!
        want_split: [logical, default = False]
            True if you want a split of part of the data
        verbose: [logical, default = False]
            True if you want to verbosely output information
    OUTPUTS:
        ax, fig -- figure axis for voxel
    '''

    ## DEFINING INDICES
    if len(grid_rgb_data.shape) == 4:
        ## ADD DIMENSION TO THE DATA
        grid_rgb_data = np.expand_dims(grid_rgb_data, axis=0) # Shape: frame, X, Y, Z, 3
        ## REDEFINING FRAME
        frame = 0
        
    ## RENORMALIZING IF NECESSARY
    if want_renormalize == True:
        grid_rgb_data = renormalize_rgb_array(grid_rgb_data)
    
    ## SEEING IF YOU WANT TO SPLIT
    if want_split == True:
        grid_rgb_data = np.split(grid_rgb_data, 2, axis = 1)[0]
    
    ## DEFINING INDICES, E.G. 1 TO 20
    r, g, b = np.indices(np.array(grid_rgb_data[frame][...,0].shape)+1)

    ## DEFINING RGB DATA TO PLOT
    grid_rgb_data_to_plot = grid_rgb_data[frame]
    
    ## PRINTING
    if verbose is True:
        print("Plotting voxels for frame %d"%(frame) )

    ## DEFINING VOXELS
    voxels = (grid_rgb_data_to_plot[...,0] > 0) | \
             (grid_rgb_data_to_plot[...,1] > 0) | \
             (grid_rgb_data_to_plot[...,2] > 0)
    
    ## DEFINING COLORS
    colors = grid_rgb_data_to_plot
    
    ## PLOTTING
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    
    ax.voxels(r, g, b ,voxels,
              facecolors=colors,
              edgecolors=np.clip(2*colors - 0.5, 0, 1),  # brighter
              linewidth=0.5) # 0.5
    
    ax.set(xlabel='x', ylabel='y', zlabel='z')
    
    plt.show()
    
    return fig, ax

### FUNCTION TO PLOT VOXEL SPLIT
def plot_voxel_split( 
               grid_rgb_data, 
               frame = 0, 
               alpha= 0.2,
               want_renormalize = False, 
               verbose = False,
               figsize = [8.0, 6.0], # Default figure size
               increment = 5,
               tick_limits = None,
               ax_limits = None,
               want_separate_axis = False,
        ):
    '''
    Function to plot voxels (3d or 2d)
    INPUTS:
        grid_rgb_data: [np.array]
            rgb data of the grid
        increment: [int]
            increments of the x/y axis
        figsize: [list]
            figure size
        verbose: [logical]
            True if you want to print
        tick_limits: [np.array, size = 3]
            limits that you would like the x, y, z dimensions to be
        ax_limits: [np.array, size = 2]
            limits that you would like the x, y, z dimensions to be
        want_separate_axis: [logical]
            True if you want figure and axis for each representation separately.
    '''
    
    ## DEFINING 2D ARRAY
    want_2D_array = False
    
    ## STORING ALPHA
    stored_alpha = alpha
    ## DEFINING INDICES
    if len(grid_rgb_data.shape) == 4: ## 3D
        ## ADD DIMENSION TO THE DATA
        grid_rgb_data = np.expand_dims(grid_rgb_data, axis=0) # Shape: frame, X, Y, Z, 3
        ## REDEFINING FRAME
        frame = 0
    elif len(grid_rgb_data.shape) == 3:  ## 2-D
        print("Since array is shape: %s"%(str(grid_rgb_data.shape) ))
        print("Printing out 2D array")
        want_2D_array = True
        
    ## RENORMALIZING IF NECESSARY
    if want_renormalize == True:
        grid_rgb_data = renormalize_rgb_array(grid_rgb_data)
    ## FINDING SHAPE
    grid_shape = grid_rgb_data.shape
    
    ## PLOTTING 2D
    if want_2D_array is True:
        ## DEFINING INDICES, E.G. 1 TO 20
        x, y= np.indices(np.array(grid_rgb_data[...,0].shape) + 1)# +1
        # z = np.zeros(x.shape)
        
        ## DEFINING VOXELS
        voxels = (grid_rgb_data[...,0] > 0) | \
                 (grid_rgb_data[...,1] > 0) | \
                 (grid_rgb_data[...,2] > 0)
                 
        ## DEFINING COLORS
        colors = grid_rgb_data
        
        ## PLOTTING
        ## CREATING FIGURE
        fig = plt.figure(figsize = figsize) 
        ax = fig.add_subplot(111)
        # fig, ax = plt.subplots(figsize = figsize)
        
        ## SETTING AXIS LABELS
        ax.set(xlabel='x', ylabel='y')
        ## PLOTTING 3D
        ax.imshow(X = grid_rgb_data, alpha = 1, aspect = 'equal')
        
        ## FINDING GRID SHAPE
        x_shape = grid_shape[0]
        y_shape = grid_shape[1]
        
        ## SETTING TICKS
        if tick_limits is None:
            tick_limits = np.arange(0,x_shape - 1, increment)
            tick_limits = np.append(arr = tick_limits,values=np.array(x_shape - 1) )
        ## SETTING X AND Y TICKS
        ax.set_xticks(tick_limits)
        ax.set_yticks(tick_limits)
        ## SETTING AXIS LIMITS
        if ax_limits is None:
            ax.set_xlim(0, x_shape -1 )
            ax.set_ylim(0, y_shape - 1)
        else:
            ax.set_xlim(ax_limits[0], ax_limits[1])
            ax.set_ylim(ax_limits[0], ax_limits[1])

    else:
        ## 3D PRINTING
        ## FINDING SHAPE OF X
        x_shape = grid_shape[1]
        y_shape = grid_shape[2]
        z_shape = grid_shape[3]
        
        ## DEFINING OFFSET
        offset_x = int(x_shape / 2.0)
        
        ## SPLITTING DATA
        split_data_grid_rgb_data = np.split(grid_rgb_data, 2, axis = 1)

        ## SEEING IF SEPARATE AXIS IS TRUE
        if want_separate_axis is True:
            ## DEFINING AXIS
            axis_types = np.arange(grid_rgb_data.shape[-1])
            ## DEFINING FIGURE AND AXIS LIST
            figs = []
            axs = []
        else:
            axis_types = [0]
            
        ## LOOPING THROUGH EACH AXIS
        for desired_axis in axis_types:
            
            ## SETTING ALPHA VALUE
            alpha = 1
            ## CREATING FIGURE
            fig = plt.figure(figsize = figsize)
            ax = fig.gca(projection='3d')
            
            ## SETTING X Y LABELS
            ax.set(xlabel='x', ylabel='y', zlabel='z')
            ## SETTING AXIS LIMITS
            if tick_limits is None:
                tick_limits = np.arange(0,x_shape + increment, increment)
            ax.set_xlim3d(0, x_shape)
            ax.set_ylim3d(0, y_shape)
            ax.set_zlim3d(0, z_shape)
            
            ## SETTING 3D AXIS
            ax.set_xticks(tick_limits)
            ax.set_yticks(tick_limits)
            ax.set_zticks(tick_limits)
            
            ## CHANGING COLOR AXIS
            if want_separate_axis is True:
                if desired_axis in [0, 1, 2]: # WATER, RED
                    color_axis = desired_axis
                    if len(axis_types) == 2: # WATER AND COSOLVENT ONLY
                        if desired_axis == 1:
                            color_axis = 2                        
                else:
                    ## GRAY
                    color_axis = None
            
            ## LOOPING THROUGH GRID DATA
            for idx, grid_rgb_data in enumerate(split_data_grid_rgb_data):
            
                ## DEFINING INDICES, E.G. 1 TO 20
                r, g, b = np.indices(np.array(grid_rgb_data[frame][...,0].shape)+1)
            
                ## DEFINING RGB DATA TO PLOT
                grid_rgb_data_to_plot = grid_rgb_data[frame]
            
                ## DEFINING VOXELS
                if want_separate_axis is not True:
                    voxels = (grid_rgb_data_to_plot[...,0] > 0) | \
                             (grid_rgb_data_to_plot[...,1] > 0) | \
                             (grid_rgb_data_to_plot[...,2] > 0)
                    ## DEFINING COLORS
                    colors = grid_rgb_data_to_plot
                else:
                    ## LOCATING VOXELS
                    voxels = (grid_rgb_data_to_plot[...,desired_axis] > 0)
                    
                    ## DEFINING GRID SHAPE
                    grid_shape = np.array(grid_rgb_data_to_plot[...,0].shape)
                    
                    ## DEFINING COLORS
                    colors = np.zeros( np.append(grid_shape,3) )
                    
                    ## DEFINING COLORS
                    if color_axis is None:
                        for each_axis in range(colors.shape[-1]):
                            colors[...,each_axis] = grid_rgb_data[...,desired_axis]
                    else:
                        ## NOT GRAY
                        colors[...,color_axis] = grid_rgb_data[...,desired_axis]

                
                if idx == 1:
                    alpha = stored_alpha
                    
                ## ADDING ALPHA VALUE
                colors = np.insert(colors, [3], [alpha], axis=3)

                ## OFFSETTING
                r = r + offset_x*idx
                
                print("Plotting alpha: %.2f"%(alpha) )
                # print(r)
                ax.voxels(r, g, b ,voxels,
                          facecolors=colors,
                          edgecolors=np.clip(2*colors - 0.5, 0, 1),  # brighter
                          linewidth=0.5,
                          alpha = alpha,
                          ) # 0.5
        
            plt.show()
            ## STORING FIGURE
            if want_separate_axis is True:
                figs.append(fig)
                axs.append(ax)
            
    ## PRINTING AS A LIST
    if want_separate_axis is True:
        fig = figs
        ax = axs
    return fig, ax

### FUNCTION TO UPDATE LIMITS
def update_ax_limits(ax, axis_ranges):
    ''' 
    This updates axis limits given the axis ranges
    INPUTS:
        ax: [obj]
            axis of your figure
        axis_ranges: [dict]
            dictionary containing axis limits (e.g. 'x', 'y'), which has a tuple containing min, max, and increment
    OUTPUT:
        ax: [obj]
            Updated axis
    '''
    ## SETTING X Y LIMS
    if axis_ranges['x'] != None:
        ax.set_xlim([axis_ranges['x'][0], axis_ranges['x'][1]])
        ax.set_xticks(np.arange( *axis_ranges['x'] ) ) # , , **AXIS_LABELS_DICT 
    
    ## SETTING Y LIMS
    if axis_ranges['y'] != None:
        ax.set_ylim([axis_ranges['y'][0], axis_ranges['y'][1]])
        ax.set_yticks(np.arange( *axis_ranges['y'] ) ) #  , **AXIS_LABELS_DICT
    
    return ax

### FUNCTION TO CHANGE AXIS LABELS
def change_axis_label_fonts( ax, labels):
    '''
    The purpose of this function is to update axis label fonts
    INPUTS:
        ax: [obj]
            axis of your figure
        labels: [dict]
            dictionary for your axis labels
    '''
    for tick in ax.get_xticklabels():
        tick.set_fontname(labels['fontname'])
        tick.set_fontsize(labels['fontsize'])
    for tick in ax.get_yticklabels():
        tick.set_fontname(labels['fontname'])
        tick.set_fontsize(labels['fontsize'])
    return ax
    
### FUNCTION TO GET CMAP
def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.
    This function is useful to generate colors between red and purple without having to specify the specific colors
    USAGE:
        ## GENERATE CMAP
        cmap = get_cmap(  len(self_assembly_coord.gold_facet_groups) )
        ## SPECIFYING THE COLOR WITHIN A FOR LOOP
        for ...
            current_group_color = cmap(idx) # colors[idx]
            run plotting functions
    '''
    ## IMPORTING FUNCTIONS
    import matplotlib.pyplot as plt
    return plt.cm.get_cmap(name, n + 1)