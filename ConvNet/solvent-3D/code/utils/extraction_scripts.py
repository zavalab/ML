#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extraction_scripts.py
This stores all extraction scripts.

FUNCTIONS:
    load_pickle: loads pickle but outputs the first result
    load_pickle_general: loads pickle and gives general result

"""
import pickle
import sys
import pandas as pd

### FUNCTION TO LOAD PICKLE
def load_pickle(Pickle_path, verbose = False):
    '''
    This function loads pickle file and outputs the first result. 
    INPUTS:
        Pickle_path: [str]
            path to the pickle file
        verbose: [logical, default = False]
            True if you want to verbosely tell you where the pickle is from
    OUTPUTS:
        results from your pickle
    '''
    # PRINTING
    if verbose == True:
        print("LOADING PICKLE FROM: %s"%(Pickle_path) )    
    ## LOADING THE DATA
    with open(Pickle_path,'rb') as f:
        # multi_traj_results = pickle.load(f) ## ENCODING LATIN 1 REQUIRED FOR PYTHON 2 USAGE
        if sys.version_info[0] > 2:
            try:    
                results = pickle.load(f, encoding='latin1') ## ENCODING LATIN 1 REQUIRED FOR PYTHON 2 USAGE
            except OSError: ## lOADING NORMALLY
                results = pickle.load(f) ## ENCODING LATIN 1 REQUIRED FOR PYTHON 2 USAGE
        elif sys.version_info[0] == 2: ## ENCODING IS NOT AVAILABLE IN PYTHON 2
            results = pickle.load(f) ## ENCODING LATIN 1 REQUIRED FOR PYTHON 2 USAGE
        else:
            print("ERROR! Your python version is not 2 or greater! We cannot load pickle files below Python 2!")
            sys.exit()
    return results[0]
            

### FUNCTION TO LOAD PICKLE
def load_pickle_general(Pickle_path, verbose = False):
    '''
    This function loads pickle file and outputs the first result. 
    INPUTS:
        Pickle_path: [str]
            path to the pickle file
        verbose: [logical, default = False]
            True if you want to verbosely tell you where the pickle is from
    OUTPUTS:
        results from your pickle
    '''
    # PRINTING
    if verbose == True:
        print("LOADING PICKLE FROM: %s"%(Pickle_path) )    
    ## LOADING THE DATA
    with open(Pickle_path,'rb') as f:
        # multi_traj_results = pickle.load(f) ## ENCODING LATIN 1 REQUIRED FOR PYTHON 2 USAGE
        if sys.version_info[0] > 2:
            try:
                results = pickle.load(f, encoding='latin1') ## ENCODING LATIN 1 REQUIRED FOR PYTHON 2 USAGE
            except ImportError: # ModuleNotFoundError or :
                results = pd.read_pickle( Pickle_path  ) # [0]
        elif sys.version_info[0] == 2: ## ENCODING IS NOT AVAILABLE IN PYTHON 2
            results = pickle.load(f) ## ENCODING LATIN 1 REQUIRED FOR PYTHON 2 USAGE
        else:
            print("ERROR! Your python version is not 2 or greater! We cannot load pickle files below Python 2!")
            sys.exit()
    return results
            