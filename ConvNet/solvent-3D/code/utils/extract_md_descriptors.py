# -*- coding: utf-8 -*-
"""
extract_md_descriptors.py
The purpose of this script is to compare the MD descriptor approach with the 
approch done by 3D CNNs

Created on: 06/21/2019

Author(s):
    - Alex K. Chew (alexkchew@gmail.com)
"""
import pandas as pd
import numpy as np
from sklearn import linear_model
from analyze_deep_cnn import analyze_deep_cnn, metrics
import os

## PLOTTING TOOLS
from read_extract_deep_cnn import plot_parity_publication_single_solvent_system

## GETTING PEARSON'S R
from scipy.stats import pearsonr

### FUNCTION TO LEAVE ONE OUT CROSS VALIDATION
def leave_one_out_cross_validation_training_testing( data ):
    '''
    The purpose of this function is to perform cross validation by 
    leave-one-out protocol. 
    INPUTS:
        data: [list]
            list of data values, e.g. 
                ['ETBE', 'tBuOH', 'PDO', 'LGA', 'FRU', 'CEL', 'XYL']
    OUTPUTS:
        cross_validation_list: [list]
            list containing permutations when leaving one out
    '''
    ## LEAVE ONE OUT CROSS VALIDATION
    cross_validation_list = []
    
    ## LOOPING THROUGH THE DATA
    for each_data in data:
        ## CREATING A COPY OF THE LIST 
        current_list = data[:]
        ## GETTING THE INDEX
        popping_index = current_list.index(each_data)
        ## NOW POPPING THE DATA
        current_list.pop(popping_index)
        ## APPENDING
        cross_validation_list.append([current_list,each_data])
    return cross_validation_list

### FUNCTION TO NORMALIZE X ARRAY
def normalize_array(X):
    '''
    The purpose of this function is to normalize an array by:
        x - min(x)/ (range(x))
    This can be used to get x values between 0 and 1. 
    INPUTS:
        X: [np.array, shape=(num_instances, num_descriptors)]
            X array containing all your data, e.g.:
                array([[-6.61814483e+00,  1.24338987e+01,  0.00000000e+00],
                       [-5.94701754e+01,  2.48893631e+00,  0.00000000e+00],
                       [-6.74026745e+01,  4.32685612e+00,  0.00000000e+00],
                       [-3.78434552e+01,  4.69077839e+00,  0.00000000e+00],
                       [ 1.01427891e-01,  5.16783217e+00,  1.70351000e-01],
                       [-2.16425333e+01,  2.67832168e+00,  1.70351000e-01],
                       [-1.85297151e+01,  1.51282051e+00,  1.70351000e-01],])
    OUTPUTS:
        
    '''
    ## FINDING MAXIMA
    max_values = np.max(X, axis = 0)
    min_values = np.min(X, axis = 0)
    ## FINDING RANGE
    range_values = max_values - min_values
    normalized_X = ( X - min_values ) / range_values
    return normalized_X

### FUNCTION TO TEST WITH MULTILINEAR MODEL
def predict_multilinear_model(regr,
                              df, 
                              molecular_descriptors,
                              output_label,
                              ):
    '''
    This function predicts with the multilinear model using the molecular 
    descriptors from training
    INPUTS:
        regr: [object]
            regression model
        df: [pd.dataframe]
            pandas dataframe object
        molecular_descriptors: [list]
            list of molecular descriptors within the dataframe
        output_label: [str]
            string of output label
    OUTPUTS:
        predicted_values: [np.array]
            array of predicted values using the input descriptors
        actual_values: [np.array]
            array of actual values
    '''
    ## EXTRACTION OF MOLECULAR DESCRIPTORS
    descriptors = np.array([ df[each_descriptor] for each_descriptor in molecular_descriptors]).T
    ## PREDICTING VALUES USING TRAINING SET
    predicted_values = regr.predict(X = descriptors)
    ## ACTUAL VALUES
    actual_values = np.array(df[output_label])
    return predicted_values, actual_values

### FUNCTION TO GENERATE MULTILINEAR MODEL
def generate_multilinear_model(df,
                               molecular_descriptors,
                               output_label,
                               normalize=True,
                               ):
    '''
    The purpose of this function is to generate a multilinear model given a dataframe, 
    molecular descriptors, and output labels. 
    INPUTS:
        df: [pd.dataframe]
            pandas dataframe object
        molecular_descriptors: [list]
            list of molecular descriptors within the dataframe
        output_label: [str]
            string of output label
        normalize: [logical, default=True]
            True if you want to normalize your X array between 0 and 1
    OUTPUTS:
        regr: [object]
            regression model
        output_data: [np.array]
            actual output data
        predicted_values: [np.array]
            array of predicted values using the input descriptors
        rmse: [float]
            root-mean-square error of the predicted vs. actual
        slope: [float]
            slope of the predicted vs. actual
        pearson_r: [float]
            pearsons' r correlation coefficient between predicted and actual
    '''
    ## EXTRACTION OF MOLECULAR DESCRIPTORS
    descriptors = np.array([ df[each_descriptor] for each_descriptor in molecular_descriptors]).T
    ## EXTRACTION OF LABELS
    output_data = np.array(df[output_label])
    ## NORMALIZING DESCRIPTORS
    if normalize is True:
        descriptors = normalize_array(X = descriptors)
    # NOTE: whether or not we normalize array, it would not change the output result! (i.e. model fitting parameters)
    # Though, it will change the weights! It is a great way to try to compare weights to see which descriptor is important
    ## GENERATING A MULTILINEAR DESCRIPTOR APPROACH
    regr = linear_model.LinearRegression()
    ## FITTING
    regr.fit( X = descriptors, y = output_data  )
    ## PREDICTING VALUES USING TRAINING SET
    predicted_values = regr.predict(X = descriptors)
    ## COMPUTING RMSE
    output_metrics = metrics(y_fit = predicted_values, y_act = output_data, want_dict = True)
    slope = output_metrics['slope']
    rmse = output_metrics['rmse']
    pearson_r = output_metrics['pearson_r']
    # _,rmse,_,_,slope = metrics(y_fit = predicted_values, y_act = output_data)
    ## COMPUTING PEARSONS R
    # pearson_r = pearsonr( x = output_data, y = predicted_values )[0]
    return regr, output_data, predicted_values, rmse, slope, pearson_r
## FUNCTION TO PLOT MODEL_STORAGE
def plot_analyzed_models( model_storage,
                          want_combined = False,
                          parity_plot_inputs = {}):
    '''
    The purpose of this function is to plot all the models.
    INPUTS:
        model_storage: [dict]
            dictionary storing all model information
        want_combined: [logical, default=False]
            True if you want to combine all the dataframes
        parity_plot_inputs: [dict]
            parity plot inputs as a dictionary
    OUTPUTS:
        plots for each of the models
    '''
    ## DEFINING EACH DATAFRAME
    data_frame_list = [ model_storage[each_model]['df'] for each_model in model_storage.keys() ]
    data_frame_keys = list(model_storage.keys())
    if want_combined is True:
        data_frame_list = [ pd.concat(data_frame_list) ]  
        data_frame_keys = [ '-'.join(data_frame_keys) ]
        
    ## LOOPING THROUGH MODEL STORAGE
    for idx, each_model in enumerate(data_frame_list):
        ## DEFINING FIGURE NAME
        if any(parity_plot_inputs):
            parity_plot_inputs['fig_name'] = list(model_storage.keys())[idx] + '.' + parity_plot_inputs['fig_extension'] 
            
        
        ## PLOTTING PARITY
        fig, ax = plot_parity_publication_single_solvent_system( dataframe = each_model,
                                                       mass_frac_water_label = 'mass_frac_water',
                                                       sigma_act_label = 'sigma_label',
                                                       sigma_pred_label = 'sigma_label_pred',
                                                       **parity_plot_inputs
                                                       )
        ## SETTING TITLE
        ax.set_title("Parity plot for: %s"%(data_frame_keys[idx]) )
    return

####################################################################
### CLASS FUNCTION TO ANALYZE REGRESSION DATA FOR MD DESCRIPTORS ###
####################################################################
class analyze_descriptor_approach:
    '''
    The purpose of this function is to analyze the descriptor approach. 
    INPUTS:
        path_md_descriptors: [str]
            path to csv file containing all molecular descriptors
        molecular_descriptors: [list]
            list of molecular descriptors
        output_label: [list]
            list of output labels
        verbose: [logical, default=False]
            True if you want everything printed out
    OUTPUTS:
        
        
    '''
    ## INITIALIZING
    def __init__(self,
                 path_md_descriptors,
                 molecular_descriptors = [ 'gamma', 'tau', 'delta' ],
                 output_label = 'sigma_label',
                 verbose = False,
                 ):
        ## INITIALIZING
        self.path_md_descriptors = path_md_descriptors
        self.molecular_descriptors = molecular_descriptors
        self.output_label = output_label
        self.verbose = verbose
        
        ## USING PANDAS TO READ FILE    
        self.csv_file = pd.read_csv( self.path_md_descriptors )
        
        return
    
    ### FUNCTION TO ANALYZE PER COSOLVENT BASIS
    def generate_multilinear_regression(self, 
                              analyze_type = 'all',
                              normalize = True, 
                              verbose = True):
        '''
        The purpose of this function is to analyze each cosolvent system individually 
        and create a model for it.
        INPUTS:
            self: [class object]
                self object
            analyze_type: [str, default='all']
                analyze type that you want, listed below:
                    'all': use all possible data
                    'cosolvent': use per cosolvent basis
            normalize: [logical, default=True]
                True if you want to normalize your X array between 0 and 1
            verbose: [logical, default=True]
                True if you want to print out model details
        OUTPUTS:
           model_storage: [dict]
               dictionary containing all model information after training / testing
        '''
        ## FINDING UNIQUE COSOLVENTS
        if analyze_type == 'cosolvent':
            unique_cosolvents = np.unique(self.csv_file['cosolvent'])
            if verbose is True:
                print("Analyzing per cosolvent basis for:")
                print(unique_cosolvents)
        else:
            unique_cosolvents = [ 'all' ]

        ## DEFINING STORAGE
        model_storage = {}
        
        ## LOOPING THROUGH EACH COSOLVENT
        for each_cosolvent in unique_cosolvents:
            if analyze_type == 'cosolvent':
                ## FINDING ONLY COSOLVENT LABELS
                df = self.csv_file[self.csv_file['cosolvent'] == each_cosolvent] # .loc
            else:
                df = self.csv_file
            
            ## GENERATING MULTILINEAR MODEL
            regr, output_data, predicted_values, rmse, slope, pearson_r = generate_multilinear_model(df = df,
                                                                                          molecular_descriptors = self.molecular_descriptors,
                                                                                          output_label = self.output_label,
                                                                                          normalize=normalize,
                                                                                          )
            ## STORING DATAFRAME
            df.insert( len(df.columns) , self.output_label + '_pred', predicted_values )
            # df[output_label + '_pred'] = pd.Series( predicted_values )
            
            ## PRINTING INTERCEPT AND COEFFICIENT
            if verbose is True:
                print("\n====== Cosolvent: %s ======="%(each_cosolvent) )
                print('Intercept: \n', regr.intercept_)
                print('Coefficients: \n', regr.coef_)
                print('Slope: %.2f'%(slope) )
                print('RMSE: %.2f'%(rmse) )
            ## STORING MODEL
            model_storage[each_cosolvent]={
                    'df': df,
                    'model': regr,
                    'actual_values'    : output_data,
                    'predicted_values' : predicted_values,
                    'coefficients'     : regr.coef_,
                    'intercept'        :  regr.intercept_,
                    'rmse'             : rmse,
                    'slope'             : slope,
                    'pearson_r'         : pearson_r,
                    }
        return model_storage
    
    ## CROSS VALIDATION
    def cross_validate( self,
                        column_name = 'cosolvent',
                        verbose = True,
                        want_overall = True,
                        ):
        '''
        The purpose of this function is to cross validate across descriptors and output.
        INPUTS:
            column_name: [str]
                name of the column you want to cross validate
            verbose: [logical, default=True]
                True if you want to print out all details
            want_overall: [logical]
                True if you want overall metrics
        OUTPUTS:
            model_storage: [dict]
                dictionary of the models
        '''
        ## FINDING ALL UNIQUE COLUMNS
        unique_columns = np.unique(self.csv_file[column_name])
        ## GENERATE CROSS VALIDATION LIST
        cross_validation_list = leave_one_out_cross_validation_training_testing(data = unique_columns.tolist())
        ## DEFINING STORAGE
        model_storage = {}
        
        ## DEFINING STORAGE FOR PREDICTED AND ACTUAL
        if want_overall is True:
            pred_values_storage = []
            act_values_storage = []
        ## LOOPING THROUGH CROSS VALIDATION LIST
        for each_cross_data in cross_validation_list:
            ## DEFININING TRAINING/TEST SET
            train_set = each_cross_data[0]
            test_set = each_cross_data[1]
            
            ## DEFINING DATAFRAMES FOR TRAINING AND TEST SET
            train_df = self.csv_file[self.csv_file[column_name].isin(train_set)]
            test_df = self.csv_file[self.csv_file[column_name].isin([test_set])]
            
            ## TRAINING THE DATA SET
            regr, output_data, predicted_values, rmse, slope, pearson_r = generate_multilinear_model(df = train_df,
                                                                                   molecular_descriptors = self.molecular_descriptors,
                                                                                   output_label = self.output_label,
                                                                                   normalize=False,
                                                                                   )
            ## PRINTING                                                   
            if verbose is True:
                print("\n====== %s: %s ======="%(column_name, ','.join(train_set)) )
                print('Intercept: \n', regr.intercept_)
                print('Coefficients: \n', regr.coef_)
                print('Train Slope: %.2f'%(slope) )
                print('Train RMSE: %.2f'%(rmse) )
            
            ## PREDICTING
            predicted_values, actual_values = predict_multilinear_model(regr,
                                                         df = test_df, 
                                                         molecular_descriptors = self.molecular_descriptors,
                                                         output_label = self.output_label,
                                                         )
            
            ## STORING DATAFRAME
            test_df.insert( len(test_df.columns) , self.output_label + '_pred', predicted_values )
            
            
            ## COMPUTING RMSE
            output_metrics = metrics(y_fit = predicted_values, y_act = actual_values, want_dict = True)
            test_rmse = output_metrics['rmse']
            test_slope = output_metrics['slope']
            
            ## STORING MODEL
            model_storage[test_set]={
                    'df': test_df,
                    'model': regr,
                    'coefficients'     : regr.coef_,
                    'intercept'        : regr.intercept_,
                    'test_rmse'        : test_rmse,
                    'test_slope'       : test_slope,
                    'pearson_r'        : pearson_r, 
                    }
            ## PRINTING
            if verbose is True:
                print("=== Test set: %s ==="%( test_set )  )
                print('Test Slope: %.2f'%(test_slope) )
                print('Test RMSE: %.2f'%(test_rmse) )
            ## STORING
            if want_overall is True:
                pred_values_storage.extend(predicted_values)
                act_values_storage.extend(actual_values)
        
        ## RUNNING ANALYSIS FOR ALL
        if want_overall is True:
            ## CONVERTING TO NUMPY
            pred_values_storage = np.array(pred_values_storage)
            act_values_storage = np.array(act_values_storage)
            ## COMPUTING METRICS
            overall_metrics = metrics(y_fit = pred_values_storage, y_act = act_values_storage, want_dict = True)
            ## OUTPUTING TO MODEL STORAGE
            model_storage['overall']={
                    'test_rmse'         : overall_metrics['rmse'],
                    'test_slope'         : overall_metrics['slope'],
                    'pearson_r'         : overall_metrics['pearson_r'],
                    }
            
        return model_storage
            
### FUNCTION TO EXTRACT NAMES AND TEST RMSE
def extract_model_storage_rmse(model_storage):
    '''
    The purpose of this function is to extract details from the model storage
    INPUTS:
        model_storage: [dict]
            model storage dictionary
    OUTPUTS:
        df: [pandas dataframe]
            dataframe containing extracted information
    '''
    ## CREATING EMPTY DICTIONARY
    extracted_dict = {}
    ## LOOPING THROUGH EACH KEY
    for each_key in model_storage:
        ## CREATING ENTRY
        extracted_dict[each_key] = {}
        ## STORING
        extracted_dict[each_key]['test_rmse'] = model_storage[each_key]['test_rmse']
        extracted_dict[each_key]['test_slope'] = model_storage[each_key]['test_slope']
        extracted_dict[each_key]['pearson_r'] = model_storage[each_key]['pearson_r']
    
    ## CREATING A DATAFRAME
    df = pd.DataFrame(extracted_dict)
    return df

#%%
## MAIN FUNCTION
if __name__ == "__main__":
    
    ## DEFINING FULL PATH TO MD DESCRIPTORS AND EXPERIMENTS
    path_md_descriptors=r"R:\scratch\3d_cnn_project\database\Experimental_Data\solvent_effects_regression_data_MD_Descriptor_with_Sigma.csv"
    
    ## DEFINING OUTPUTP ATH
    path_output = r"C:\Users\akchew\Box Sync\VanLehnGroup\0.Manuscripts\Solvent_effects_3D_CNNs\Excel_Spreadsheet\csv_md_descriptors"
    ## OUTPUT FOR COSOLVENT
    path_output_cosolvent_cross = os.path.join( path_output, r"cross_valid_cosolvents.csv" )
    path_output_solute_cross = os.path.join( path_output, r"cross_valid_solutes.csv" )
    
    ## USING PANDAS TO READ FILE    
    # csv_file = pd.read_csv( path_md_descriptors )
    ## DEFINING INPUTS
    inputs={ 
             'path_md_descriptors': path_md_descriptors,
             'molecular_descriptors' : [ 'gamma', 'tau', 'delta' ],
             'output_label' : 'sigma_label',
             'verbose' : False,
            }
    
    ## LOADING DESCRIPTORS
    analyzed_descriptors = analyze_descriptor_approach(**inputs)

    ## RUNNING PER COSOLVENT TRAINING
    model_storage = analyzed_descriptors.generate_multilinear_regression(
                                    # analyze_type = 'all',
                                    analyze_type = 'cosolvent',
                                    normalize=True,
                                    verbose = True,            
                                    )
    #%%
    ## DEFINING FIGURE SIZE
    figure_size=( 18.542/3, 18.542/3 )
    
    ## DEFINING PARITY PLOT INPUTS    
    parity_plot_inputs = \
        {
                'save_fig_size': figure_size,
                'save_fig': True,
                'fig_extension': 'svg',
                }   
    
    ## PLOTTING MODELS
    plot_analyzed_models( model_storage = model_storage,
                          parity_plot_inputs = parity_plot_inputs)
    
    #%%
    
    ## RUNNING CROSS VALIDATION
    cross_valid_cosolvent = analyzed_descriptors.cross_validate(
                            column_name='cosolvent',
                            want_overall = True)    
    
    ## EXTRACTING DATA
    data_extract = extract_model_storage_rmse(model_storage = cross_valid_cosolvent,
                                              )
    ## PRINTING
    data_extract.to_csv(path_output_cosolvent_cross)
    
    #%%
    ## PLOTTING MODELS
#    plot_analyzed_models( cross_valid_cosolvent,
#                          want_combined = True)
    
    #%%
    
    ## RUNNING CROSS VALIDATION
    cross_valid_solute = analyzed_descriptors.cross_validate(
                            column_name='solute',
                            want_overall = True)
    ## EXTRACTING DATA
    data_extract = extract_model_storage_rmse(model_storage = cross_valid_solute)
    ## PRINTING
    data_extract.to_csv(path_output_solute_cross)
    ## PLOTTING MODELS
#    plot_analyzed_models( cross_valid_solute,
#                          want_combined = True)
    
        
    
