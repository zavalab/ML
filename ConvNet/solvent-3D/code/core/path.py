# -*- coding: utf-8 -*-
"""
path.py
This script contains all path information. In particular, database paths, csv paths, etc.
We will create command to first check the user. Then, we will use a class function 
to get the paths for that particular user. 

Created by: Alex K. Chew (alexkchew@gmail.com, 04/17/2019)

FUNCTIONS:
    find_user: function to locate the user
    find_paths: function to find paths
    read_combined_name_directories: reads combined names
    extract_combined_names_to_vars: function to extract the combined names from 'read_combined_name_directories'
    

"""
## IMPORTING MODULES
import sys
import time
import os
import glob

## IMPORTING NOMENCLATURE
from core.nomenclature import read_combined_name, extract_representation_inputs, extract_sampling_inputs

### FUNCTION TO FIND THE USER
def find_user():
    '''
    The purpose of this function is to find the current user information. 
    This is necessary to automate the finding of paths.
    INPUTS:
        null
    OUTPUTS:
        user_info: [dict]
            dictionary containing user information
    '''
    import getpass
    import os
    ## USERNAME
    username = getpass.getuser()
    ## HOME DIRETCTORY
    homedir = os.environ['HOME']
    user_info = {
            'username': username,
            'homedir': homedir,
            }
    return user_info
    
### FUNCTION TO DELINEATE PATH INFORMATION
def find_paths(skip_error=True):
    '''
    The purpose of this function is to find the paths for databases and 
    test classification details. This code will first look for the user, then decide 
    if you are in the database. If so, define the paths appropriately.
    INPUTS:
        null -- we will find paths based on your user information
    OUTPUTS:
        path_dict: [dict]
            dictionary containing all path information
    '''
    ## START BY FINDING USERS
    user_info = find_user()
    
    ## THEN, DELINEATE CSV FILE INFORMATION, ETC.
    if user_info['username'] == 'akchew' and user_info['homedir'] == 'C:\\Users\\akchew':
        database_path = r"C:\Users\akchew\Box Sync\2019_Spring\CS760\Spring_2019_CS760_Project\Datasets\20_20_20_Gridded_Data"
        csv_path=r"C:\Users\akchew\Box Sync\2019_Spring\CS760\Spring_2019_CS760_Project\Datasets\Experimental_Data\solvent_effects_regression_data.csv"
        combined_database_path=r"C:\Users\akchew\Box Sync\2019_Spring\CS760\Spring_2019_CS760_Project\Combined_dataset"
        output_path=r"C:\Users\akchew\Box Sync\2019_Spring\CS760\Spring_2019_CS760_Project\Output"
        result_path=r"C:\Users\akchew\Box Sync\2019_Spring\CS760\Spring_2019_CS760_Project\Results"
    elif user_info['username'] == 'alex' and user_info['homedir'] == '/Users/alex':
        database_path = r"/Users/alex/Box Sync/2019_Spring/CS760/Spring_2019_CS760_Project/Datasets/20_20_20_Gridded_Data"
        csv_path=r"/Users/alex/Box Sync/2019_Spring/CS760/Spring_2019_CS760_Project/Datasets/Experimental_Data/solvent_effects_regression_data.csv"
        combined_database_path=r"/Users/alex/Box Sync/2019_Spring/CS760/Spring_2019_CS760_Project/Combined_dataset"
        output_path=r"/Users/alex/Box Sync/2019_Spring/CS760/Spring_2019_CS760_Project/Output"
        result_path=r"/Users/alex/Box Sync/2019_Spring/CS760/Spring_2019_CS760_Project/Results"
    
    else:
        if skip_error == False:
            print("Error! Username and home directory currently undefined!")
            print("Current username: %s"%( user_info['username'] ) )
            print("Current home directory: %s"%( user_info['homedir'] ) )
            print("Please update core > path.py to include your username and home directory")
            print("Pausing here for 5 seconds before termination ...")
            time.sleep(5)
            sys.exit()
        else:
            print("-----------------------------------------------------")
            print("Username (%s) and home directory (%s) not defined!"%( user_info['username'], user_info['homedir'] ) )  
            print("Turning off find paths code in core > path.py...")
            print("We are now assuming you have given paths. If this is not true, you may receive errors!")
            print("-----------------------------------------------------")
            ## SKIPPING ERROR MESSAGES
            database_path, combined_database_path, csv_path, output_path, result_path = None, None, None, None, None
        
    ## DEFINING A PATH DICTIONARY
    path_dict = {
            'database_path': database_path,
            'combined_database_path': combined_database_path,
            'csv_path': csv_path,
            'output_path': output_path,
            'result_path': result_path
            }
    return path_dict
    

### FUNCTION TO READ ALL DIRECTORIES USING READ_COMBINED_NAME COMMAND
def read_combined_name_directories(path, 
                                   extraction_func = None, 
                                   want_dir = True,
                                   want_single_path = False):
    '''
    The purpose of this function is to read all directory names within a path
    and extract via combined name. 
    INPUTS:
        path: [str]
            path to your directories
        extraction_func: [func]
            function to extract the names of a single basename directory
        want_single_path: [str]
            True if you want a single path, not a list of paths
    OUTPUTS:
        directory_paths: [list]
            directory paths
        directory_basename: [list]
            list of directory basenames
        directory_extracted_names: [list]
            list of extracted names
    '''
    ## FINDING ALL DIRECTORIES
    if want_single_path is False:
        directory_paths = glob.glob( os.path.join(path ,'*') ) 
    else:
        directory_paths = [path]

    ## DIRECTORIES ONLY
    if want_dir is True:
        directory_paths = [a for a in directory_paths if os.path.isdir(a)]
    
    ## FINDING ALL DIRECTORY NAMES
    directory_basename = [ os.path.basename(os.path.normpath(each_directory))
                                        for each_directory in directory_paths]
    
    try:
        if extraction_func is None:
            ## EXTRACTED NAMES
            directory_extracted_names = [ read_combined_name(each_directory)
                                                for each_directory in directory_basename]
        else:
            ## EXTRACTED NAMES
            directory_extracted_names = [ extraction_func(each_directory)
                                                for each_directory in directory_basename]
    except IndexError:
        print("Error! Extraction protocol did not succeed. Please check your representation")
        print("Current paths: ")
        print(directory_paths)
        print("Current basename: ")
        print(directory_basename)
        sys.exit(1)
        
    
    return directory_paths, directory_basename, directory_extracted_names
    
## EXTRACTION OF THE STRING
def extract_input_MD_name( input_string ):
    '''
    The purpose of this function is to extract the input MD trajectory name, e.g. 'FRU_393.15_ACE_12'
    INPUTS:
        input_string: [str]
            input string
    OUTPUTS:
        output_dict: [dict]
            dictionary which contains the input string as a dict
    '''
    ## SPLITTING INPUT STRING
    input_string_split = input_string.split('_')
    
    ## CHECKING GVL COSOLVENT
    cosolvent = input_string_split[2]
    mass_frac=input_string_split[3]
    if cosolvent == "GVL":
        cosolvent="GVL_L"
        mass_frac=input_string_split[4]    
    
    ## CREATING DICTIONARY
    output_dict = {
            'solute': input_string_split[0],
            'temp': input_string_split[1],
            'cosolvent': cosolvent,
            'mass_frac': mass_frac,
            }
    return output_dict

#### FUNCTION TO EXTRACT READ_COMBINED_NAMES TO VARIABLES
def extract_combined_names_to_vars(extracted_name, want_dict=False):
    '''
    The purpose of this function is to extract combined names to variables. 
    Essentially, we extract information from "read_combined_names" code.
    INPUTS:
        extracted_name: [dict]
            dictionary of extracted name. 
            e.g.:
             {'cnn_type': 'solvent_net',
              'data_type': '20_20_20_50ns',
              'epochs': '500',
              'mass_frac_data': ['10', '25', '50', '75'],
              'representation_inputs': '5_0.05_5000',
              'representation_type': 'split_avg_nonorm_perc',
              'solute_list': ['CEL', 'ETBE', 'FRU', 'LGA', 'PDO', 'XYL', 'tBuOH'],
              'solvent_list': ['DIO', 'GVL', 'THF']}]
        want_dict: [logical, default=False]
            True if you want dictionary as an output
    OUTPUTS:
        if want_dict == False:
            representation_type: [str]
                representation type
            representation_inputs: [dict]
                representation type as a dictionary
            data_type: [str]
                type of data that you are inputting
            num_epochs: [int]
                number of epochs
            solute_list: [list]
                list of solutes
            solvent_list: [list]
                list of solvents
            mass_frac_data: [list]
                list of mass fraction data
        else:
            extract_rep: [dict]
                dictionary containing all extraction information
    '''
    ## EXTRACTION OF DETAILS FROM THE DIRECTORY NAME
    representation_type = extracted_name['representation_type']    
    representation_inputs = extract_representation_inputs( representation_type = representation_type, 
                                                           representation_inputs = extracted_name['representation_inputs'].split('_') )
    
    ## UPDATING SAMPLING INPUTS
    sampling_dict = extract_sampling_inputs( sampling_type = extracted_name['sampling_type'], 
                                             sampling_inputs = extracted_name['sampling_inputs'].split('_'),)
            
    data_type = extracted_name['data_type']
    cnn_type  = extracted_name['cnn_type']
    ## REDEFINIG NUMBER OF EPOCHS    
    num_epochs = extracted_name['epochs']
    
    ## DEFINING SOLUTE LIST, ETC.
    solute_list = extracted_name['solute_list']
    solvent_list = extracted_name['solvent_list']
    mass_frac_data = extracted_name['mass_frac_data']
    
    ## DEFINING WANT DESCRIPTORS
    want_descriptor = extracted_name['want_descriptor']
    
    if want_dict is False:
        return representation_type, \
                representation_inputs, \
                sampling_dict, \
                data_type, \
                cnn_type, \
                num_epochs, \
                solute_list, \
                solvent_list, \
                mass_frac_data, \
                want_descriptor,
    else:
        ## DEFINING DICTIONARY
        extract_rep = {
                'representation_type' : representation_type,
                'representation_inputs' : representation_inputs,
                'sampling_dict' : sampling_dict,
                'data_type' : data_type,
                'cnn_type' : cnn_type,
                'num_epochs' : num_epochs,
                'solute_list': solute_list,
                'solvent_list' : solvent_list,
                'mass_frac_data': mass_frac_data,
                'want_descriptor': want_descriptor,
                }
        return extract_rep
