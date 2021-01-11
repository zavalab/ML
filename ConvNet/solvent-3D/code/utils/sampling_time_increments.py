# -*- coding: utf-8 -*-
"""
sampling_time_increments.py
The purpose of this script is to find the increments required to correctly 
minimize the RMSE. The underlying goal is to find the total amount of increment time 
average to get the same RMSE (without having to use extensive amounts of data). 

The algorithm is as follows:
    - Input the solute, etc.
    - Input the time increments desired, e.g. 500,1000, .... and so on. These time 
    increments are necessary to figure out how much data is needed. 
    - Use a bash script to analyze the increments, etc.

Created on: 05/24/2019

Author(s):
    - Alex K. Chew (alexkchew@gmail.com)
"""
## IMPORTING MODULES
import os
import time
import sys

## IMPORTING GLOBAL VARIABLES
from core.global_vars import SOLUTE_TO_TEMP_DICT
## CHECKING TOOLS
from core.check_tools import check_testing
## IMPORTING ML FUNCTIONS
from core.ml_funcs import get_list_args

#####################################################
### CLASS FUNCTION TO DO SAMPLING TIME INCREMENTS ###
#####################################################
class sampling_time_increments:
    '''
    The purpose of this function is to generate sampling time increment code 
    that could be used 
    '''




#%%
## MAIN FUNCTION
if __name__ == "__main__":
    
    ## SEE IF TESTING IS ON
    testing = check_testing()
    
    ## SEEING IF TESTING IS TRUE
    if testing == True:
        ## DEFINING SOLVENT LIST
        solvent_list = [ 'DIO', 'GVL', 'THF' ]# 'GVL', 'THF' ] # , 'GVL', 'THF' 
        ## DEFINING MASS FRACTION DATA
        mass_frac_data = ['10', '25', '50', '75'] # , '25', '50', '75'
        ## DEFINING SOLUTE LIST
        solute_list = list(SOLUTE_TO_TEMP_DICT)
        
        ## DEFINING SAMPLING INCREMENTS INPUTS
        total_sampling_time_ps=50000 # ps
        increments_ps_per_frame=10 # ps per frame
        
        ## DEFINING DESIRED SPLITTING
        num_splits=5
        
        ## DEFINING DESIRED FRAMES
        desired_frames=[500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
        
    
    
    
    
    
        
        
        