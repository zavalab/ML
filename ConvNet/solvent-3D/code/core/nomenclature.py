# -*- coding: utf-8 -*-
"""
nomenclature.py
This script contains all nomenclature information

FUNCTIONS:
    convert_to_single_name: converts to a single name
    extract_instance_names: extracts the instance name
    extract_representation_inputs: exract represnetation inputs as a dictionary
    ## FOR TRAINING
    get_combined_name: gets combined names
    read_combined_name: reverses the get combined name function
    
    
"""

### IMPORING IMPORTANT MODULES
import sys

### FUNCTION TO CONVERT SOLUTES, ETC. TO NAME
def convert_to_single_name( solute, solvent, mass_fraction, temp ):
    ''' This function converts multiple arguments to a single name '''
    ## DEFINING ORDER
    order = [ solute, temp, solvent, mass_fraction ]
    str_name = '_'.join(order)
    return str_name


### FUNCTION TO EXTRACT NAMES
def extract_instance_names( name ):
    '''
    The purpose of this function is to extract instance names
    INPUTS:
        name: [str]
            name of your instance, e.g. 'tBuOH_363.15_DIO_10'
    OUTPUTS:
        name_dict: [dict]
            name dictionary
    '''
    ## SPLITTING
    split_name = name.split('_')
    
    ## DEFINING NAME DICTIONARY
    name_dict ={
            'solute': split_name[0],    # Solute name
            'temp': split_name[1],      # Temperature in K   
            'cosolvent': split_name[2], # Cosolvent name
            'mass_frac': split_name[3], # Mass fraction of water
            }
    return name_dict

### FUNCTION THAT TAKES REPRESENTATION INPUTS BASED ON TYPE
def extract_representation_inputs(representation_type, representation_inputs):
    '''
    The purpose of this function is to extract representation inputs based on type. 
    For example, 'split_avg_nonorm_perc' has three inputs:
        num splits, percentage, and total frames
    We would like to extract the inputs correctly.
    INPUTS:
        representation_type: [str]
            representation type that we are interested in
        representation_inputs: [list]
            list of representation inputs
    OUTPUTS:
        representation_inputs_dict: [dict]
            representation inputs as a dictionary
        
    '''
    ## FIXING REPRESENTATION INPUTS
    if representation_type != 'split_avg_nonorm_perc':
        representation_inputs_dict = {
                                    'num_splits': int(representation_inputs[0])
                                    }
        if representation_type == 'split_avg_nonorm_sampling_times':
            representation_inputs_dict = {
                                        'num_splits': int(representation_inputs[0]),
                                        'perc': float(representation_inputs[1]),
                                        'initial_frame': int(representation_inputs[2]),
                                        'last_frame': int(representation_inputs[3]),
                                        }
        
    else:
        representation_inputs_dict = {
                                    'num_splits': int(representation_inputs[0]),
                                    'perc': float(representation_inputs[1]),
                                    'total_frames': int(representation_inputs[2]),
                                    }
    return representation_inputs_dict

### FUNCTION TO EXTRACT SAMPLING INPUTS BASED ON TYPE
def extract_sampling_inputs( sampling_type, 
                             sampling_inputs,):
    '''
    The purpose of this function is to extract the sampling inputs into a format 
    that is understandable. The sampling information is output into the training 
    algorithm.
    
    Available sampling types:
        strlearn: 
            stratified learning (by default), allowing you to split training and test sets
        spec_train_tests_split: 
            way to optimize your number of trianing and testing splits. We assume that 
            the training and test sets are selected from the end of the trajectory, where 
            the last N_test is the test set and N_train is the training set. 
    
    
    INPUTS:
        sampling_type: [str]
            sampling type that you are trying to use
        sampling_inputs: [list]
            sampling inputs
    OUTPUTS:
        sampling_dict: [dict]
            dictionary for sampling
    '''
    ## STORING THE NAME
    sampling_dict = {
            'name': sampling_type,
            }
    
    ## DEFINING AVAILABLE SAMPLING DICTS
    available_sampling_dict = [ 'strlearn', 'spec_train_tests_split',  ]
    
    ## DEFINING LEARNING TYPE
    if sampling_type == 'strlearn':
        sampling_dict['split_percentage'] =  float(sampling_inputs[0])
    elif sampling_type == 'spec_train_tests_split':
        sampling_dict['num_training'] = int(sampling_inputs[0])
        sampling_dict['num_testing'] = int(sampling_inputs[1])
    else:
        print("Error! sampling_type is not correctly defined. Please check the 'extract_sampling_inputs' function to ensure your sampling dictionary is specified!")
        print("Pausing here so you can see the error!")
        print("Available sampling types are:")
        print("%s"%(', '.join( available_sampling_dict ) ) )
        sys.exit(1)
    return sampling_dict
    

### FUNCTION TO DECIDE THE NAME
def get_combined_name(representation_type,
                      representation_inputs,
                      solute_list,
                      solvent_list,
                      mass_frac_data,
                      data_type = "20_20_20",
                      ):
    '''
    The purpose of this function is to combine all the names into a single 
    framework that we can store files in. 
    INPUTS:
        representation_type: [str]
            string of representation types
        representation_inputs: [dict]
            dictionary for the representation input
        solute_list: [list]
            list of solutes you are interested in
        solvent_list: [list]
            list of solvent data, e.g. [ 'DIO', 'GVL', 'THF' ]
        mass_frac_data: [list]
            list of mass fraction data, e.g. ['10', '25', '50', '75']
    OUTPUTS:
        unique_name: [str]
            unique name characterizing all of this
    '''
    ## SORING SOLUTE NAMES
    solute_list.sort()
    ## SORTING COSOLVENT NAMES
    solvent_list.sort()
    ## SORTING MASS FRACTION INFORMATION
    solvent_list.sort()
    ## SORT REPRESENTATION INPUTS AS A LIST
    representation_inputs_list = [ str(representation_inputs[each_key]) for each_key in sorted(representation_inputs) ]
    unique_name =   data_type + '-' + \
                    representation_type + '-' + \
                    '_'.join(representation_inputs_list) + '-' + \
                    '_'.join(solute_list) + '-' + \
                    '_'.join(solvent_list) + '-' + \
                    '_'.join(mass_frac_data)
    return unique_name

### FUNCTION TO READ COMBINED NAME
def read_combined_name(unique_name, reading_type = "post_training"):
    ''' 
    The purpose of this function is to go from combined name back to 
    representative inputs. 
    INPUTS:
        unique_name: [str], e.g:
            20_20_20_100ns_updated-split_avg_nonorm_sampling_times-10_0.1_0_10000-spec_train_tests_split-1_2-solvent_net-500-CEL_ETBE_FRU_LGA_PDO_XYL_tBuOH-10_25_50_75-DIO_GVL_THF
            
            20_20_20_100ns_updated-
            split_avg_nonorm_sampling_times-
            10_0.1_0_10000-
            spec_train_tests_split-
            1_2-
            solvent_net-
            500-
            CEL_ETBE_FRU_LGA_PDO_XYL_tBuOH-10_25_50_75-DIO_GVL_THF
            
            unique name characterizing all of this
        reading_type: [str, default='post_training']
            type to read, e.g
                post_training: 
                    post training examples
                instances:
                    combined training instances
                
    OUTPUTS:
        combined_name_info: [dict]
            dictionary with the combined names revived
    '''
    ## DEFINING EMPTY 
    combined_name_info = {}
    
    ## SPLITTING
    split_name = unique_name.split('-')
    
    ## EXTRACTION
    if reading_type == 'post_training':
        combined_name_info['data_type'] = split_name[0]
        combined_name_info['representation_type'] = split_name[1]
        combined_name_info['representation_inputs'] = split_name[2]
        combined_name_info['sampling_type'] = split_name[3]
        combined_name_info['sampling_inputs'] = split_name[4]
        
        combined_name_info['cnn_type'] = split_name[5]
        combined_name_info['epochs'] = split_name[6]
        
        combined_name_info['solute_list'] = split_name[7].split('_')
        combined_name_info['mass_frac_data'] = split_name[8].split('_')
        combined_name_info['solvent_list'] = split_name[9].split('_')
    elif reading_type == 'instances':
        combined_name_info['data_type'] = split_name[0]
        combined_name_info['representation_type'] = split_name[1]
        combined_name_info['representation_inputs'] = split_name[2] # .split('_')
        combined_name_info['solute_list'] = split_name[3].split('_')
        combined_name_info['solvent_list'] = split_name[4].split('_')
        combined_name_info['mass_frac_data'] = split_name[5].split('_')
    else:
        print("Error, no reading type found for: %s"%(reading_type  ) )
        print("Check read_combined_name code in core > nomenclature")
        sys.exit()
    ## APPENDING IF TRUE
    if split_name[-1] == "MD":
        combined_name_info['want_descriptor'] = True
    else:
        combined_name_info['want_descriptor'] = False
    
    return combined_name_info
