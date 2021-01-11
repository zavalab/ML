# -*- coding: utf-8 -*-
"""
read_extract_deep_cnn.py
The purpose of this script is to read "extract_deep_cnn.py"

Created on: 04/25/2019

Author(s):
    - Alex K. Chew (alexkchew@gmail.com)
"""
## IMPORTING OS
import os
import pandas as pd
import glob
import matplotlib.pyplot as plt
import itertools
import numpy as np

## IMPORTING ANALYSIS TOOL
# from analyze_deep_cnn import analyze_deep_cnn

## TAKING EXTRACTION SCRIPTS
from extraction_scripts import load_pickle_general

## IMPORTING MODULES
import core.plotting_scripts as plotter

from analyze_deep_cnn import metrics
from matplotlib.offsetbox import AnchoredText

## IMPORTING READING TOOLS
from core.nomenclature import read_combined_name
## IMPORTING PATHS
from core.path import extract_combined_names_to_vars

### FUNCTION MAKE PLOT
def plot_parity_publication_single_solvent_system( dataframe,
                                                   fig_suffix = "_MD_Predicted",
                                                   fig_extension = "svg",
                                                   mass_frac_water_label='mass_frac_water',
                                                   sigma_act_label='sigma_label',
                                                   sigma_pred_label = 'sigma_pred',
                                                   sigma_pred_err_label = None,
                                                   fig_size_cm = (17.1, 17.1),
                                                   save_fig_size = (16.8/3, 16.8/3),
                                                   x_lims = (-1.5, 2.5, 1 ),
                                                   y_lims = (-1.5, 2.5, 1 ),
                                                   cross_validation_training_info = None,
                                                   cross_validation_training_info_stored = None,
                                                   want_multiple_cosolvents = False,
                                                   want_exp_error = True,
                                                   fig_name = None,
                                                   save_fig = False,
                                                   fig = None,
                                                   ax = None):
    '''
    The purpose of this plot is to plot publication ready parity 
    plots. We have different labels for different reactnats and different products. 
    INPUTS:
        dataframe: [pandas obj]
            data for your system, e.g.
               solute cosolvent  mass_frac_water  sigma_label  sigma_pred
            54   ETBE       THF             0.10        -0.41   -0.319189
        save_fig: [logical, default=False]
            True if you want to save the figure
        sigma_label: [str, default=sigma_label]
            sigma label within your data frame
        sigma_pred_err: [str, default = None]
            prediction error. If None, no error bar is plotted.
        cross_validation_training_info: [dict]
            dictionary containing cross validation information
        cross_validation_training_info_stored: [dict]
            dictionary containin cross validation details
        want_multiple_cosolvents: [logical, default=False]
            False if you do not want to enforce multiple cosolvents
        sigma_pred: [str, default=sigma_pred]
            sigma prediction within your data frame
        x_lims: [tuple]
            x limits
        y_lims: [tuple]
            y limits
        save_fig_size: [tuple]
            figure size in cm
        fig/ax: [fig, default = None]
            figure object, if None, then this is recreated
        want_exp_error: [logical, default=True]
            True if you want experimental error
    OUTPUTS:
        fig, ax: figure, axis for your plot
    '''
    ## FINDING COSOLVENT NAME (FIRST INSTANCE)
    cosolvent_name = dataframe['cosolvent'].iloc[0]
    
    ## SEEING IF MORE THAN ONE COSOLVENT IS PRESENT
    unique_cosolvent_names = np.unique(dataframe.cosolvent)
    if len(unique_cosolvent_names) > 1:
        multiple_cosolvent = True
        ## REDEFINING COSOLVENT NAME
        cosolvent_name =   '_'.join(unique_cosolvent_names.tolist()) #  np.array2string( unique_cosolvent_names )
    else:
        multiple_cosolvent = False
    
    
    ## IF YOU WANT MULTIPLE COSOLVENTS
    if want_multiple_cosolvents is True:
        multiple_cosolvent = True
    
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
    
    ## CHECKING IF CROSS VALIDATION IS DONE
    if cross_validation_training_info is None:
        ## GETTING PRED VALUES
        y_true = np.array(dataframe[sigma_act_label])
        y_pred = np.array(dataframe[sigma_pred_label])
        
        ## COMPUTING RMSE
        mae, rmse, evs, r2, slope, pearson_r = metrics( y_fit = y_pred,y_act = y_true )

    else:
        rmse = cross_validation_training_info['training_rmse']
        slope = cross_validation_training_info['training_slope']
        test_rmse = cross_validation_training_info['test_rmse']
            
    ## DEFINING FIGURE
    if fig is None or ax is None:
        fig, ax = plotter.create_fig_based_on_cm( fig_size_cm = fig_size_cm )
    
        ## SETTING AXIS LABELS
        ax.set_xlabel("sigma_act")
        ax.set_ylabel("sigma_pred")
    
    ## CHECKING IF MASS FRAC > 1
    if float(dataframe[mass_frac_water_label].max()) > 1:
        divide_by_100 = True
    else:
        divide_by_100 = False
    
    ## GETTING SIGMA
    for idx, row in dataframe.iterrows():
        ## DEFINING VARIABLES
        solute = row['solute']
        
        ## SEEING IF WE NEED TO DIVIDE BY 100
        if divide_by_100 is False:
            mass_frac_water = "%.2f"%(row[mass_frac_water_label])
        else:
            ## DIVIDING BY 100
            mass_frac_water = "%.2f"%( float(row[mass_frac_water_label])/100.)
        ## SIGMA
        sigma_act = row[sigma_act_label]
        sigma_pred = row[sigma_pred_label]
        
        ## DEFINING COLOR
        color = plotter.SOLUTE_COLOR_DICT[solute]
        label = solute
        marker = plotter.MASS_FRAC_SYMBOLS_DICT[mass_frac_water]
        
        ## DEFINING COSOLVENT FILL COLOR
        if multiple_cosolvent == True:
            ## FINDING THE CURRENT COSOLVENT
            cosolvent = row['cosolvent']
            ## FINDING FILL STYLE
            fillstyle = plotter.COSOLVENT_FILL_DICT[cosolvent]                
        else:
            fillstyle = 'full'
        
        ## DEFINING ALPHA VALUE
        alpha = 1 # default
        
        ## SEEING IF WE NEED TO DIM SOME OF THE CROSS VALIDATION INFORMATION
        if cross_validation_training_info is not None:
            ## DEFINING DEFAULT CHANGE ALPHA
            change_alpha = False
            if cross_validation_training_info['cross_validation_name'] == 'solute':
                if cross_validation_training_info['test_set_variables'] == 'tBuOH':
                    test_solute_name = 'TBA'
                else:
                    test_solute_name = cross_validation_training_info['test_set_variables']
                if solute != test_solute_name:
                    change_alpha = True
            elif cross_validation_training_info['cross_validation_name'] == 'mass_frac':
                if row['mass_frac'] != cross_validation_training_info['test_set_variables']:
                    change_alpha = True
            elif cross_validation_training_info['cross_validation_name'] == 'cosolvent':
                if row['cosolvent'] != cross_validation_training_info['test_set_variables']:
                    change_alpha = True
            if change_alpha is True:
                alpha = cross_validation_training_info['alpha']
            else:
                alpha = 1

        ## PLOTTING ERROR BARS
        if sigma_pred_err_label is not None:        
            ## GETTING THE ERROR
            yerr = row[sigma_pred_err_label]
            ## PLOT POINTS WITH ERROR BAR
            ax.errorbar( sigma_act, 
                         sigma_pred, 
                         yerr = yerr, 
                         color = color, 
                         linewidth=1,
                         fmt = marker, 
                         fillstyle = fillstyle, 
                         capsize=2,
                         alpha = alpha,) # linestyle="None" elinewidth=3, markeredgewidth=10 
        else:
            
            ## PLOT POINTS WITHOUT ERROR BAR
            ax.plot(sigma_act, 
                       sigma_pred, 
                       marker = marker, 
                       color=color, 
                       linewidth=1, 
                       fillstyle = fillstyle,
                       alpha = alpha,
                       label = label + '_' + mass_frac_water,
                       **plotter.PLOT_INFO) # label = "Max tree depth: %s"%(max_depth)
    
    ##############################
    ### CHANGING FIGURE LIMITS ###
    ##############################
    
    # ---- SETTING X AND Y LIMITS ---- $
    # DEFINING LIMITS
    # x_lims = (-1.5, 2.5, 1 ) # 0.5
    # y_lims = (-1.5, 2.5, 1 ) # 0.5
    
    ## SETTING X TICKS AND Y TICKS
    ax.set_xticks(np.arange(x_lims[0], x_lims[1] + x_lims[2], x_lims[2]))
    ax.set_yticks(np.arange(y_lims[0], y_lims[1] + y_lims[2], y_lims[2]))
    # ---- SETTING X AND Y LIMITS ---- $
    
    ## DRAWING X AND Y AXIS AT ZERO
    ax.axhline(y=0, linestyle='--', linewidth=0.5, color='gray')
    ax.axvline(x=0, linestyle='--', linewidth=0.5, color='gray')
    
    ## DRAWING X AND Y AXIS
    lims = np.array([
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ])
    ## PLOTTING 
    ax.plot(lims, lims, 'k-', linewidth = 1, alpha=1.00, zorder=0)
    
    ## PLOT EXPERIMENTAL ERROR
    if want_exp_error is True:
        exp_error = 0.10
        lower_lim = lims[0]
        upper_lim = lims[1]
        points_x = [lower_lim + exp_error , lower_lim ]
        points_y = [ upper_lim, upper_lim - exp_error ]
        combined_points = np.array([points_x, points_y]) # lower axis
        
        ## PLOTTING LOWER AXIS
        ax.plot(combined_points[:,0], combined_points[:,1], 'k--', linewidth = 0.5, alpha=1.00, zorder=0)
        ## PLOTTING UPPER AXIS
        ax.plot(combined_points[:,1], combined_points[:,0], 'k--', linewidth = 0.5, alpha=1.00, zorder=0)
        
    ax.set_aspect('equal')
    ax.set_xlim([x_lims[0], x_lims[1]] )
    ax.set_ylim([y_lims[0], y_lims[1]])
#    ax.set_xlim(lims)
#    ax.set_ylim(lims)
    
    #########################
    ### CREATING BOX TEXT ###
    #########################
    if cross_validation_training_info is not None or cross_validation_training_info_stored is not None:
        ## SEEING IF STORAGE DONE
        if cross_validation_training_info_stored is not None:
            if cross_validation_training_info_stored['last_one'] is True:
                ## LOOPING THROUGH EACH DATA
                box_text = ''
                for idx, each_info in enumerate(cross_validation_training_info_stored['data']):
                    if idx != len(cross_validation_training_info_stored['data'])-1:
                        box_text = box_text + "%s: %.2f\n"%( each_info['test_set_variables'] + " RMSE", each_info['test_rmse'],
                                                  )
                    else:
                        box_text = box_text + "%s: %.2f"%( each_info['test_set_variables'] + " RMSE", each_info['test_rmse'],
                                                  )
        else:
            box_text = "%s: %.2f\n%s: %.2f\n%s: %.2f"%( "Train slope", slope,
                                              "Train RMSE", rmse,
                                              "Test RMSE", test_rmse,
                                              ) 
            
    else:
        box_text = "%s: %.2f\n%s: %.2f"%( "Slope", slope,
                                          "RMSE", rmse) 

    ## ADDING TEXT BOX
    try:
        print(box_text)
        box_text_available = True
    except NameError:
        box_text_available = False
        
    ## DRAWING ON PLOT
    if box_text_available is True:
        text_box = AnchoredText(box_text, frameon=True, loc=4, pad=0.5)
        plt.setp(text_box.patch, facecolor='white', alpha=0.5)
        ax.add_artist(text_box)
    
    ## SHOW TIGHT LAYOUT
    fig.tight_layout()
    
    ## SETTING FIGURE SIZE
    if save_fig_size is not None:
        fig.set_size_inches(plotter.cm2inch( *save_fig_size ))
        ## SHOW TIGHT LAYOUT
        fig.tight_layout() # ADDED TO FIT THE CURRENT LAYOUT
    else:
        # SETTING AXIS TO SCALED
        ax.axis('scaled')
    
    ## STORING FIGURE
    if save_fig == True:
        if fig_name is None:
            fig_name = cosolvent_name + '_' + fig_suffix + '.' + fig_extension
        print("Printing figure: %s"%(fig_name) )
        fig.savefig( fig_name, 
                     format=fig_extension, 
                     dpi = 1200,    )
    return fig, ax

### FUNCTION TO RUN ANALYSIS
def run_analysis(analysis, 
                 results_pickle_file, 
                 image_file_path, 
                 csv_file_path='.', 
                 print_csv=True, 
                 save_fig = False,
                 want_fig=True):
    ''' 
    This function runs the analysis
    INPUTS:
        want_fig: [logical, default=True]
            True if you want the figures plotted
    '''


    ## DEFINING OUTPUT FILE NAME
    output_file_name = results_pickle_file
    # output_file_name, extension =  os.path.splitext(results_pickle_file)

    ## PRINTING
    print("Working on %s"%(output_file_name) )

    if want_fig is True:
        ## PLOTTING LEARNING CURVE
        fig, ax = analysis.plot_learning_curve(loss_type="loss", 
                                               fig_name = os.path.join( image_file_path, output_file_name + "-training_loss-learning_curve" ) ,
                                               save_fig=save_fig, 
                                               fig_format = "png") # val_mean_squared_error
        
        ## VALIDATION MEAN SQUARED ERROR
        fig, ax = analysis.plot_learning_curve(loss_type="val_loss", 
                                               fig_name = os.path.join( image_file_path, output_file_name + "-validation_loss-learning_curve" ) ,
                                               save_fig=save_fig, 
                                               fig_format = "png") # val_mean_squared_error
    #    ## PLOTTING PARITY SCATTER
    #    fig, ax = analysis.plot_parity_scatter(save_fig = save_fig, 
    #                                           fig_name = output_file_name + "-parity_plot" ,
    #                                           fig_format = "png")

    ## DEFINING RMSE and SLOPE
    rmse = analysis.accuracy_dict['rmse']
    slope = analysis.accuracy_dict['slope']
    
    ## ANALYSIS DICTIONARY
    print("RMSE: %.15f"%(rmse))
    print("Best-fit Slope: %.15f"%(slope))

    ## Y PRED, Y ACT, Y_ERR
#        y_true = analysis.y_true # X axis
#        y_pred = analysis.y_pred # Y axis
#        y_pred_std = analysis.y_pred_std # Y error
    
    
    df = analysis.dataframe # dataframe.set_index('solute')
    
    ## PLOTTING PARITY
    if want_fig is True:
        fig, ax = plot_parity_publication_single_solvent_system( dataframe = df,
                                                       fig_suffix = "-parity_cosolvent_plot_colored",
                                                       fig_name = os.path.join( image_file_path, output_file_name + "-parity_cosolvent_plot_colored.png" ),
                                                       fig_extension = "png",
                                                       mass_frac_water_label = 'mass_frac',
                                                       sigma_act_label = 'y_true',
                                                       sigma_pred_label = 'y_pred',
                                                       sigma_pred_err_label = 'y_pred_std',
                                                       save_fig_size = (8, 8),
                                                       save_fig = save_fig)
    
    
    ## COSOLVENT SPECIFIC PLOTS
#    fig, ax = analysis.plot_parity_scatter_cosolvent(save_fig = save_fig, 
#                                       fig_name = os.path.join( image_file_path, output_file_name + "-parity_cosolvent_plot"),
#                                       fig_format = "png")

    ## PRINTING DATAFRAME
    dataframe = analysis.dataframe
    
    ## DATAFRAME TO CSV
    if print_csv is True:
        dataframe.to_csv( os.path.join( csv_file_path, output_file_name + "_parity_plot.csv" )  )

    ## ADDING COSOLVENT
    accuracy_info = pd.DataFrame(analysis.cosolvent_regression_accuracy)
    ## ADDING ALL SOLVENT
    accuracy_info['all'] = pd.Series(analysis.accuracy_dict)
    ## DATAFRAME TO CSV
    if print_csv is True:
        accuracy_info.to_csv( os.path.join( csv_file_path, output_file_name + "_rmse.csv" )  )
    
    ## DEFINING TIME IN SECONDS
    total_training_time = plotter.convert_hms_to_Seconds(analysis.time_hms)
    
    return total_training_time, rmse

### FUNCTION TO GENERATE DATAFRAME
def generate_dataframe_slope_rmse( sim_path,
                                   list_of_directories = None,
                                   results_pickle_file = 'model.results',
                                   desired_row_label = 'cnn_type',
                                   desired_network = None,
                                   ):
    '''
    The purpose of this function is to generate dataframe of slope and rmse.
    INPUTS:
        sim_path: [str]
            simulation path
        list_of_directories: [list]
            list of directories for simulations
        results_pickle_file: [str]
            results pickle file to look within directories
        desired_row_label: [str, default='cnn_type']
            desired row label. If none, entire simulation name will be used
        desired_network: [str, default = None]
            desired network
    OUTPUTS:
        df: [pandas]
            pandas dataframe containing slope and RMSE, e.g.:
                             DIO_GVL_THF_rmse  DIO_GVL_THF_slope  ...  THF_rmse  THF_slope
                orion                0.285623           0.766212  ...  0.229974   0.822497
                solvent_net          0.113954           0.963795  ...  0.078002   0.975581
                voxnet               0.207022           0.868555  ...  0.158509   0.923775
    '''
    ## GLOBBING ALL DIRECTORIES
    directory_list = [ glob.glob(os.path.join(sim_path,each_directory,'*')) for each_directory in list_of_directories ]

    ## JOINING LIST
    directory_list_flatten = list(itertools.chain.from_iterable(directory_list))
    
    ## CREATING EMPTY DATAFRAME
    df = pd.DataFrame()
    
    ## CREATING EMPTY DICTIONARY
    database_dict = {}
    
    ## LOOPING THROUGH DIRECTORY LIST
    for main_sim_path in directory_list_flatten:
        ## GETTING SIMULATION NAME
        simulation_name = os.path.basename(main_sim_path)
        ## EXTRACTING
        current_directory_extracted = read_combined_name(simulation_name)
        ## EXTRACTING INFORMATION
        extract_rep = extract_combined_names_to_vars(extracted_name = current_directory_extracted,
                                                     want_dict = True)
        
        ## DEFINING ROW LABEL
        if desired_row_label is not None:
            row_label = extract_rep[desired_row_label]
        else:
            row_label = simulation_name
#        representation_type, \
#        representation_inputs, \
#        sampling_dict, \
#        data_type, \
#        cnn_type, \
#        num_epochs, \
#        solute_list, \
#        solvent_list, \
#        mass_frac_data,\
#        want_descriptors= extract_combined_names_to_vars(extracted_name = current_directory_extracted)
        
        ## DEFINING NETWORK
        if desired_network is None:
            ## DEFINING ADD TO DICTIONARY
            add_to_dict = True
        else:
            cnn_type = extract_rep['cnn_type']
            if cnn_type == desired_network:
                add_to_dict = True
            else:
                add_to_dict = False
        
        ## ADDING TO DICTIONARY
        if add_to_dict is True:
            ## COMBINING SOLVENT LIST
            solvent_list_combined = '_'.join(extract_rep['solvent_list'])
            
            ## DEFINING FULL PATH
            results_full_path = os.path.join( main_sim_path, results_pickle_file )
            
            ## DEFINING ANALYSIS
            analysis = pd.read_pickle( results_full_path  )[0]
            
            ## DEFINING ANALYSIS DICTIONARY
            analysis_dict = analysis.accuracy_dict
            
            ## DEFINING SLOPE AND RMSE
            slope = analysis_dict['slope']
            rmse = analysis_dict['rmse']
            
            ## SEEING IF KEYS ARE ADDED
            if row_label not in database_dict.keys():
                ## ADDDING TO KEYS
                database_dict[row_label] = {}
            
            ## ADDING TO DICT
            database_dict[row_label][solvent_list_combined+'_slope'] = slope
            database_dict[row_label][solvent_list_combined+'_rmse'] = rmse

    ## CONVERTING DICT TO PANDAS
    df = pd.DataFrame.from_dict(database_dict).transpose()
    return df


#%%
## MAIN FUNCTION
if __name__ == "__main__":
    
    ## DEFINING MAIN DIRECTORY
    simulation_dir = r"190617-all_solvent_training" # Single solvent 190617-each_solvent_training
    simulation_dir = r"190617-each_solvent_training"
    simulation_dir = r"190623-20_20_20ns_withoxy_SolventNet"
    simulation_dir = r"190701-16_16_16_all_solvents" # 16 x 16 x 16
    simulation_dir = r"190705-20_20_20_20ns_firstwith10_each_solvent"
    simulation_dir = r"190725-newrep_20_20_20_20ns_oxy_3chan"
    simulation_dir = r"190731-parity_20_20_20_20ns_solvent_only_firstwith10"
    simulation_dir = r"190731-parity_20_20_20_20ns_3channel_hydroxyl_firstwith10"
    simulation_dir = r"190801-parity_20_20_20_20ns_4chan_hydroxyl_firstwith10"
    main_dir = os.path.join(r"R:\scratch\3d_cnn_project\simulations", simulation_dir)
    
            
    ## DEFINING CSV PATH
    csv_file_path = r"R:\scratch\3d_cnn_project\csv_output"
    
    ## DEFINING IMAGE FILE PATH
    image_file_path = r"R:\scratch\3d_cnn_project\images"
    
    ## DEFINING RESULTS PICKLE
    results_pickle_file = r"model.results" # DIO 
    
    ## SAVING FIGURE ?
    save_fig = True
    
    ## PRINTING CSV?
    print_csv = False
    
    ## GETTING LIST OF DIRECTORIES
    directory_list = glob.glob(  os.path.join(main_dir,'*'))
    
    ## LOOPING THROUGH EACH DIRECTORY
    for main_sim_path in directory_list:
        ## GETTING SIMULATION NAME
        simulation_name = os.path.basename(main_sim_path)
        
        ## DEFINING FULL PATH
        results_full_path = os.path.join( main_sim_path, results_pickle_file )

        ## DEFINING ANALYSIS
        analysis = pd.read_pickle( results_full_path  )[0]
        # analysis = load_pickle_general( results_full_path  )[0]
        
        ## RUNNING ANALYSIS
        run_analysis( 
                      analysis = analysis,
                      results_pickle_file = simulation_name,
                      csv_file_path = csv_file_path,
                      image_file_path = image_file_path,
                      print_csv = print_csv,
                      save_fig = save_fig,
                     )
            