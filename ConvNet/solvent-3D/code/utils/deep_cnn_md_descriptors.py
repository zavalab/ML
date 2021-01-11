# -*- coding: utf-8 -*-
"""
deep_cnn_md_descriptors.py
The purpose of this code is to use molecular descriptors and neural networks 
to predict experimental reaction rates

Created on: 06/25/2019

Author(s):
    - Alex K. Chew (alexkchew@gmail.com)
"""
## IMPORTING CORE MODULES
import os
import pandas as pd
import numpy as np

## IMPORTING NOMENCLATURE
from core.nomenclature import read_combined_name, extract_representation_inputs, extract_instance_names
## IMPORTING PATHS
from core.path import read_combined_name_directories, extract_combined_names_to_vars, extract_input_MD_name
## IMPORTING COMBINING ARRAYS
from combining_arrays import combine_instances

## FUNCTION TO LOAD DESCRIPTORS
from core.global_vars import DEFAULT_PATH_DICT, INPUTS_FOR_DESCRIPTOR_FXN

## FUNCTION TO RENAME
from core.database_scripts import rename_df_column_entries, find_index_matching_columns_for_df, fix_TBA_entries

## FUNCTION TO GET THE MIN MAX SCALER
from sklearn.preprocessing import MinMaxScaler
    
## IMPORTING KERAS MODELS
from keras.models import Model
from keras.layers import Input, Dense

### FUNCTION TO GET NUMERICAL INPUT
class get_descriptor_inputs:
    '''
    The purpose of this function is to go through the instance list and extract 
    desired molecular descriptors. 
    INPUTS:
        instance_list: [list]
            instances list, e.g. 
             'LGA_403.15_THF_75',
             'PDO_433.15_DIO_10',
        path_csv: [str]
            path to csv file where all descriptors are stored
        col_names: [list]
            list of columns that you want descriptors to be inputted        
        col_matching_list: [list]
            matching columns between instances and the new list
        index_to_csv: [str,default='index_to_csv']
            string that is included into database
    OUTPUTS:
        self.dfs: [list]
            list of databases containing original instances and molecular descriptor database
        self.csv_file: [object]
            csv file object that contains all the information
        self.output_dfs: [database]
            database with molecular descriptors (no normalization)
        self.output_dfs_normalized: [database]
            renormalized dfs with molecular descriptors
    '''
    ## INITIALIZING
    def __init__(self,
                 instance_list,
                 path_csv,
                 col_names = [ 'gamma', 'tau', 'delta' ],
                 col_matching_list = [['solute','cosolvent','mass_frac'],
                                      ['solute','cosolvent','mass_frac_water']
                                      ],
                 index_to_csv = 'index_to_csv',
                 ):
        ## STORING INPUTS
        self.instance_list = instance_list
        self.path_csv = path_csv
        self.col_names = col_names
        self.index_to_csv = index_to_csv
        ## EXTRACT INSTANCES
        self.extract_instances()
        
        ## LOADING CSV FILE
        self.csv_file = pd.read_csv( self.path_csv )
        
        ## FIXING TBA ENTRIES
        self.csv_file = fix_TBA_entries( df = self.csv_file )
        
        ## MATCHING DATAFRAMES
        self.dfs = find_index_matching_columns_for_df( dfs = [ self.instances_df, self.csv_file ], 
                                                  cols_list = col_matching_list,
                                                  index_to_list_col_name = self.index_to_csv,
                                                  )
        
        ## NOW, GOING THROUGH AND ADDING EACH COLUMN NAME
        self.extract_match_dfs()
        
        ## NORMALIZING
        self.normalize_dfs()
        
        return
    
    ### FUNCTION TO EXTRACT INSTANCES
    def extract_instances(self):
        ''' This function extracts the instances and renames any entries as necessary ''' 
        ## GENERATING EXTRACTED INSTANCE LIST
        self.instances_df= pd.DataFrame([ extract_instance_names(each_instance) \
                                         for each_instance in self.instance_list ])
        
        ## FIXING TBA ENTRIES
        self.instance_df = fix_TBA_entries( df = self.instances_df)
        ## CONVERTING COLUMNS TO NUMERIC
        # self.instances_df["mass_frac"] = pd.to_numeric(self.instances_df["mass_frac"])
        return

    ### FUNCTION TO EXTRACT INFORMATION FROM DEF
    def extract_match_dfs(self):
        ''' 
        This function matches the df and outputs the corresponding gamma, etc.
        OUTPUTS:
            self.output_dfs: [dataframe]
                dataframe with the desired columns 
        '''
        ## GETTING INDEX
        index_to_cols = np.array(self.dfs[0][self.index_to_csv])
        
        ## LOOPING THROUGH COLS
        for each_col in self.col_names:
            ## CREATE EMPTY COLS
            self.dfs[0][each_col] = np.array(self.dfs[1][each_col][index_to_cols])
        ## DEFINING OUTPUT DF
        self.output_dfs = self.dfs[0]
        return
    
    ### FUNCTION TO RENORMALIZE DF
    def normalize_dfs(self):
        '''
        This function normalizes the dataframe
        To transform:
            testContinuous = self.min_max_scalar.transform(test[continuous])
        '''
        ## CREATING A MIN MAX SCALER
        self.min_max_scalar = MinMaxScaler()
        ## CREATING RENORMALIZED DF
        self.output_dfs_normalized = self.output_dfs.copy()
        ## GETTING MOLECULAR DESCRIPTORS
        df_descriptors = self.output_dfs[self.col_names]
        ## TRANSFORMING
        self.train_renormalized = self.min_max_scalar.fit_transform( np.array(df_descriptors) ) 
        
        ## STORING
        self.output_dfs_normalized[self.col_names] = self.train_renormalized[:]
        
        return
    ### FUNCTION TO TRANSFORM NEW DFS
    def transform_test_df(self, df):
        '''
        The purpose of this function is to transform a test df
        INPUTS:
            df: [database]
                database containing molecular descriptors
        OUTPUTS:
            test_output_df: [database]
                renormalized test output dfs
        ## TESTING
        test_output_df = descriptor_inputs.transform_test_df(df = descriptor_inputs.output_df)
        '''
        ## GETTING ALL DESCRIPTORS
        df_descriptors = df[self.col_names]
        ## CREATING A COPY OF DF
        test_output_df = df.copy()
        ## TRANSFORMING
        renormalized_values = self.min_max_scalar.transform( np.array(df_descriptors) )
        ## STORING
        test_output_df[self.col_names] = renormalized_values[:]
        return test_output_df
    
## MD DESCCRIPTORS
def md_descriptor_network(dim, regress=True):
    '''
    Model for VoxNet. 
    INPUTS:
        dim: [int]
            dimension of the network
        regress: [logical, default=True]
            True if you want your model to have a linear regression at the end
    OUTPUT:
        model: [obj]
            tensorflow model
    '''
    ## INPUT LAYER
    input_layer = Input(shape=(dim,) )
    
    ## ADDING OUTPUT LAYER IF DESIRED
    if regress is True:
        ## CREATING LINEAR MODEL
        output_layer = Dense(units=1, activation='linear')(input_layer)
    else:
        output_layer = input_layer
    
    ## DEFINE MODEL WITH INPUT AND OUTPUT LAYERS
    model = Model(inputs=input_layer, outputs=output_layer)    

    return model
    

#%%
## MAIN FUNCTION
if __name__ == "__main__":

    ## DEFINING INSTANCE PATH
    instance_name = r"20_20_20_32ns_first-split_avg_nonorm-8-CEL_ETBE_FRU_LGA_PDO_XYL_tBuOH-DIO_GVL_THF-10_25_50_75"
    path_instance = os.path.join(r"R:\scratch\3d_cnn_project\combined_data_set", instance_name)
    
    ## EXTRACTING DIRECTORY INFORMATION
    current_directory_extracted = read_combined_name( instance_name, reading_type = 'instances' )
        
    ## UPDATING REPRESATION INPUTS
    representation_inputs = extract_representation_inputs( representation_type = current_directory_extracted['representation_type'], 
                                                           representation_inputs = current_directory_extracted['representation_inputs'] )
    
    ## LOADING THE DATA
    instances = combine_instances(
                     solute_list = current_directory_extracted['solute_list'],
                     representation_type = current_directory_extracted['representation_type'],
                     representation_inputs = representation_inputs,
                     solvent_list = current_directory_extracted['solvent_list'], 
                     mass_frac_data = current_directory_extracted['mass_frac_data'], 
                     verbose = True,
                     database_path = DEFAULT_PATH_DICT['database_path'],
                     class_file_path = DEFAULT_PATH_DICT['class_file_path'],
                     combined_database_path = DEFAULT_PATH_DICT['combined_database_path'],
                     data_type = current_directory_extracted['data_type'],
                     )
    
    ## PRINTING NAME
    print(instances.instance_names)
    
    #%%    
    
    ## DEFINING INPUTS FOR DESCRIPTOR FUNCTION
    inputs_for_descriptor_fxn=INPUTS_FOR_DESCRIPTOR_FXN.copy() # {**INPUTS_FOR_DESCRIPTOR_FXN}
    inputs_for_descriptor_fxn['instance_list'] = instances.instance_names
    
    ## DEFINING DESCRIPTOR INPUTS
    descriptor_inputs = get_descriptor_inputs(**inputs_for_descriptor_fxn)
    
    #%%
    # TRANSFORMING
    # test_output_df = descriptor_inputs.transform_test_df(df = descriptor_inputs.output_dfs)

    