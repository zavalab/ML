# -*- coding: utf-8 -*-
"""
database_scripts.py
The purpose of this script is to contain all dataframe code

FUNCTIONS:
    rename_df_column_entries: renames column entries in a dataframe
    find_index_matching_columns_for_df: function that matches column indices for multiple dfs
    fix_TBA_entries: function to fix TBA labels
    
Created on: 06/25/2019

Author(s):
    Alex K. Chew (alexkchew@gmail.com)

"""
import pandas as pd
import numpy as np

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

### FUNCTION TO MATCH AND CREATE AN INDEX LIST
def find_index_matching_columns_for_df( dfs, 
                                        cols_list,
                                        index_to_list_col_name = 'index_to_csv'
                                        ):
    '''
    The purpose of this function is to find the index matching between columns. 
    INPUTS:
        dfs: [list]
            list of dfs. 1st one is the reference. 2nd is the one we are looking at. 
        cols_list: [list]
            list of matching instances. For example, suppose df 1 has 'solute','cosolvent','mass_frac' and 
            df 2 has 'solute','cosolvent','mass_frac_water'. You will match with a list of list:
                col_list = [ ['solute','cosolvent','mass_frac'],
                             ['solute','cosolvent','mass_frac_water'] ]
        index_to_list_col_name: [str, default='index_to_csv']
    OUTPUTS:
        dfs: [list]
            same dfs, except df 1 has a new column based on index_to_list_col_name name, which can 
            be used to reference df 2. Check out dfs[0]['index_to_csv']
    '''
    ## DEFINING DFS
    instances_df = dfs[0]
    csv_file = dfs[1]
    
    ## DEFINIGN COLUMN LISTS
    cols_instances = cols_list[0]
    cols_csv_file = cols_list[1]
    
    ## ADDING EMPTY COLUMN OF NANS
    instances_df["index_to_csv"] = np.nan
    column_index = instances_df.columns.get_loc("index_to_csv")
    ## FINDING LOCATING LABELS
    locating_labels_csv_file = np.array([ csv_file[each_label] for each_label in cols_csv_file]).T.astype('str')
    
    ## CREATING INDEX LIST
    index_list = []
    
    ## LOOPING THROUGH EACH INSTANCE
    for index, row in instances_df.iterrows():
        ## FINDING RESULTS
        current_labels = np.array([ row[each_col] for each_col in cols_instances ]).astype('str')
        ## FINDING INDEX
        try:
            index_to_csv = int(np.argwhere( (locating_labels_csv_file == current_labels).all(axis=1) )[0][0])
        except IndexError:
            print("Error found in label:", current_labels) 
        ## APPENDING INDEX
        instances_df.iloc[index, column_index] = index_to_csv
        index_list.append(index)
        
    ## CONVERTING TO INT
    instances_df["index_to_csv"] = instances_df["index_to_csv"].astype('int')
    
    return dfs

## FUNCTION TO FIX TBA ENTRIES
def fix_TBA_entries(df, col_name = 'solute', change_col_list = [ 'tBuOH', 'TBA'  ]):
    ''' This fixes TBA entries that are incorrectly labeled as tBuOH'''
    ## RENAMING TBA ENTRIES
    if df['solute'].str.match('tBuOH').any() == True:
        df = rename_df_column_entries(df = df,
                                      col_name = col_name,
                                      change_col_list = change_col_list,
                                      )
    
    return df