#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_tools.py
This script contains all checking tools.

Written by: Alex K. Chew (02/17/2019)

"""
## IMPORTING MODUELS
import sys
import os

### FUNCTION TO SEE IF TESTING SHOULD BE TURNED ON
def check_testing():
    '''
    The purpose of this function is to turn on testing if on SPYDER
    INPUTS:
        void
    OUTPUTS:
        True or False depending if you are on the server
    '''
    ## CHECKING PATH IF IN SERVER
    # if sys.prefix != '/Users/alex/anaconda' and sys.prefix != r'C:\Users\akchew\AppData\Local\Continuum\Anaconda3' and sys.prefix != r'C:\Users\akchew\AppData\Local\Continuum\Anaconda3\envs\py35_mat': 
    if any('SPYDER' in name for name in os.environ):
        print("*** TESTING MODE IS ON ***")
        testing = True
    else:
        testing = False
    return testing

### FUNCTION TO GET THE PATH ON THE SERVER
def check_path_to_server(path):
    '''
    The purpose of this function is to change the path of analysis based on the current operating system. 
    Inputs:
        path: Path to analysis directory
    Outputs:
        path (Corrected)
    '''
    ## IMPORTING MODULES
    import getpass
    
    ## CHANGING BACK SLASHES TO FORWARD SLASHES
    backSlash2Forward = path.replace('\\','/')
    
    ## CHECKING PATH IF IN SERVER
    if sys.prefix == '/usr' or sys.prefix == '/home/akchew/envs/cs760': # At server
        ## CHECKING THE USER NAME
        user_name = getpass.getuser() # Outputs $USER, e.g. akchew, or bdallin
        
        # Changing R: to /home/akchew
        path = backSlash2Forward.replace(r'R:','/home/' + user_name)
    
    ## AT THE MAC
    elif '/Users/' in sys.prefix:
        ## LISTING ALL VOLUMES
        volumes_list = os.listdir("/Volumes")
        ## LOOPING UNTIL WE FIND THE CORRECT ONE
        final_user_name =[each_volume for each_volume in volumes_list if 'akchew' in each_volume ][-1]
        ## CHANGING R: to /Volumes
        path = backSlash2Forward.replace(r'R:','/Volumes/' + final_user_name)
    
    ## OTHERWISE, WE ARE ON PC -- NO CHANGES
    
    return path

### FUNCTION TO CHECK MULTIPLE PATHS
def check_multiple_paths( *paths ):
    ''' 
    Function that checks multiple paths
    INPUTS:
        *paths: any number of paths        
    OUTPUTS:
        correct_path: [list]
            list of corrected paths
    '''
    correct_path = []
    ## LOOPING THROUGH
    for each_path in paths:
        ## CORRECTING
        correct_path.append(check_path_to_server(each_path))
    
    ## CONVERTING TO TUPLE
    correct_path = tuple(correct_path)
    return correct_path
