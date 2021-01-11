# -*- coding: utf-8 -*-
"""
ml_funcs.py
This contains functions related to machine learning

Created by: Alex K. Chew (alexkchew@gmail.com, 04/17/2019)

FUNCTIONS:
    locate_test_instance_value: 
        Locates test instance given a csv file
    get_list_args: 
        get list arguments for input data
    get_split_index_of_list_based_on_percentage: 
        functions to split index of a list based on a percentage

"""

## GETTING PEARSON'S R
from scipy.stats import pearsonr
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import numpy as np


### FUNCTION TO LOCATE TEST INSTANCE VALUE
def locate_test_instance_value( csv_file, solute, cosolvent, mass_frac_water, temp = None  ):
    '''
    This code locates the test instance class value.
    INPUTS:
        csv_file: [object]
            csv file
        solute: [str]
            name of the solute
        cosolvent: [str]
            name of the cosolvent
        mass_frac_water: [str]
            mass fraction of water
    OUTPUTS:
        value: [str]
            positive or negative for class value
    '''
#    print(solute,cosolvent,mass_frac_water)
    try:
        if temp is None:
            location = csv_file.loc[(csv_file['solute'] == solute) & (csv_file['cosolvent'] == cosolvent) & (csv_file['mass_frac_water'] == int(mass_frac_water)), 'sigma_label']
        else:
            location = csv_file.loc[(csv_file['solute'] == solute) & \
                                    (csv_file['cosolvent'] == cosolvent) & \
                                    (csv_file['mass_frac_water'] == int(mass_frac_water)) & \
                                    (csv_file['temp'] == float(temp) ),
                                    'sigma_label'
                                    ]
        ## CONVERTING LOCATION TO A STRING
        value = list(location)[0]
    except IndexError:
        value = 'nan' # Turning nans since we did not find the correct label
    return value

## CALL FUNCTION TO CONVERT TO LIST
def get_list_args(option, opt_str, value, parser):
    setattr(parser.values, option.dest, value.split(','))
    
    
### FUNCTION THAT RETURNS INDEX OF A SPLIT LIST
def get_split_index_of_list_based_on_percentage( input_list, split_percentage ):
    '''
    The purpose of this function is to get a split index of a list based on 
    percentage. This is done by rounding and using the percentage to multiply the 
    length of the list.
    INPUTS:
        input_list: [list]
            list that you want to split
        split_percentage: [float]
            percentage that you want, must be less than or equal to 1
    OUTPUTS:
        split_index: [int]
            split index for your list
    '''
    split_index = int(round(split_percentage*len(input_list)))
    return split_index


## COMPUTING ROOT MEAN SQUARED ERROR
def compute_rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


## COMPUTING BEST FIT SLOPE
def best_fit_slope_and_intercept(xs,ys):
    # Reference: https://pythonprogramming.net/how-to-program-best-fit-line-machine-learning-tutorial/?completed=/how-to-program-best-fit-line-slope-machine-learning-tutorial/
    m = (((np.mean(xs)*np.mean(ys)) - np.mean(xs*ys)) /
         ((np.mean(xs)*np.mean(xs)) - np.mean(xs*xs)))
    
    b = np.mean(ys) - m*np.mean(xs)
    
    return m, b


## FUNCTION TO CALCUALTE MSE, R2, EVS, MAE
def metrics(y_fit,y_act, want_dict = False):
    '''
    The purpose of this function is to compute metrics given predicted and actual
    values.
    INPUTS:
        y_fit: [np.array]
            fitted values as an array
        y_act: [np.array]
            actual values
        want_dict: [logical, default=False]
            True if you want the output to be a dictionary instead of a tuple
    OUTPUTS:
        mae: [float]
            mean averaged error
        rmse: [float]
            root mean squared errors
        evs: [float]
            explained variance score
        r2: [float]
            R squared for linear fit
        slope: [float]
            best-fit slope between predicted and actual values
        pearson_r_value: [float]
            pearsons r
        
    '''
    ## EXPLAINED VARIANCE SCORE
    evs = explained_variance_score(y_act, y_fit)
    ## MEAN AVERAGE ERROR
    mae = mean_absolute_error(y_act, y_fit)
    ## ROOT MEAN SQUARED ERROR
    rmse = compute_rmse(predictions=y_fit, targets = y_act) # mean_squared_error(y_act, y_fit)
    ## SLOPE AND INTERCEPT
    slope, b = best_fit_slope_and_intercept( xs = y_act, ys = y_fit)
    ## R-SQUARED VALUE
    r2 = r2_score(y_act, y_fit)
    ## PEARSON R
    pearson_r = pearsonr( x = y_act, y = y_fit )[0]
    if want_dict is False:
        return mae, rmse, evs, r2, slope, pearson_r
    else:
        ## CREATING DICTIONARY
        output_dict = {
                'evs': evs,
                'mae': mae,
                'rmse': rmse,
                'slope': slope,
                'r2': r2,
                'pearson_r': pearson_r
                }
        ## OUTPUTING
        return output_dict
