# -*- coding: utf-8 -*-
"""
cross_validation.py
The purpose of this script is to generate jobs for cross validation. We will use 
already available training deep cnn code, train them for a subset of our data, then 
generate test it on the subset that is withheld. 

This script will simply generate the different permuations of reactant/cosolvent 
that is required to correctly cross-validate. Then, we will use a bash script to 
loop through all possible systems, train them, then run post analysis by reloading 
the weights. Note that this is optimal since we can run multiple Keras instances 
across multiple cores -- whereas python alone cannot well-parallelize. 

Created on: 05/15/2019

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


### FUNCTION TO LEAVE ONE OUT CROSS VALIDATION
def leave_one_out_cross_validation( data ):
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
        cross_validation_list.append(current_list)

    return cross_validation_list

########################################################
### CLASS FUNCTION TO GENERATE CROSS VALIDATION FILE ###
########################################################
class cross_validation:
    '''
    The purpose of this code is to generate permutations for cross-validation 
    procedures. 
    INPUTS:
        input_variables: [dict]
            input dictionary containing lists that you want to vary
        cross_validation_name: [str, default = solute]
            cross validation name
        cross_validation_type: [str, default=leave_one_out]
            cross validation type
        output_path: [str, default='cross_valid.txt']
            path to output the cross validation file. 
    OUTPUTS:
        self.available_cross_valid: [list]
            available cross validation types
        self.cross_validation_list: [list]
            cross validation list with permutations
        
    '''
    ## INITIALIZING
    def __init__(self, 
                 input_variables,
                 cross_validation_name = "solute",
                 cross_validation_type = "leave_one_out",
                 output_path = 'cross_valid.txt',
                 ):
        ## STORING INITIAL VARIABLES
        self.input_variables = input_variables
        self.cross_validation_name = cross_validation_name
        self.cross_validation_type =cross_validation_type
        self.output_path = output_path
        
        ## DEFINING AVAILABLE CROSS VALIDATINO TYPE
        self.available_cross_valid = [
                'leave_one_out' # leave one out validation type
                ]
        
        ## TESTING IF INPUT VARIABLES IS WITHIN THE CROSS VALIDATION TYPE
        if self.cross_validation_name not in self.input_variables.keys():
            print("Error! Cross validation type is not within input variables!")
            print("Cross validation input: %s"%(self.cross_validation_name ))
            print("Available input variables: %s"%( ', '.join( self.input_variables.keys()) ) )
            print("Stopping here to prevent further errors!")
            time.sleep(5)
            sys.exit(1)
            
        ## GENERATING CROSS VALIDATION LIST
        if self.cross_validation_type == "leave_one_out":
            self.cross_validation_list = leave_one_out_cross_validation(data = self.input_variables[self.cross_validation_name])
        else:
            print("Error! Cross validation type is not defined: %s"%(self.cross_validation_type) )
            print("Available cross validation types: %s"%( ', '.join(self.available_cross_valid) ) )
            print("Stopping here to prevent further errors!")
            time.sleep(5)
            sys.exit(1)
        
        ## FINDING TOTAL CROSS VALIDATION
        self.total_cross_validation = len(self.cross_validation_list)
        
        ## WRITING TO FILE
        self.output_cross_validation_file()
        
        return
    
    ### FUNCTION TO OUTPUT CROSS VALIDATION FILE
    def output_cross_validation_file(self):
        '''
        The purpose of this script is to output the cross validation file.
        '''
        ## FINDING ALL KEYS THAT ARE NOT THE VALIDATION NAME
        keys_not_varied = [ each_key for each_key in self.input_variables.keys() if each_key != self.cross_validation_name ]
        
        ## FINDING BASENAME OF THE PATH
        file_name = os.path.basename(self.output_path)
        
        ## DEFINING HEADER
        header="%s\n\n"%(file_name)
        
        ## OPENING FILE
        with open( self.output_path, 'w' ) as file:
            ## START WITH HEADER
            file.write(header)
            
            file.write("--- VARIABLES_NOT_VARIED_START ---\n")
            ## WRITING VARIABLES NOT VARIED
            for each_key in keys_not_varied:
                ## WRITING
                file.write("%s %s\n"%(each_key, ','.join(self.input_variables[each_key])   ) )
            ## ADDING EMPTY SPACE
            file.write("--- VARIABLES_NOT_VARIED_END ---\n")
            file.write("\n")
            ## WRITING CROSS VALIDATION
            file.write("CROSS_VALIDATION_NAME: %s\n"%( self.cross_validation_name )  )
            file.write("CROSS_VALIDATION_TOTAL: %d\n"%( self.total_cross_validation )  )
            file.write("--- CROSS_VALIDATION_START ---\n")
            ## LOOPING THROUGH EACH CROSS VALIDATION
            for each_cross_validation in self.cross_validation_list:
                file.write("%s\n"%( ",".join(each_cross_validation) ) )
            
            file.write("--- CROSS_VALIDATION_END ---\n")
        return
        

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
        ## DEFINING CROSS VALIDATION INPUTS
        cross_validation_name = "solute"
        cross_validation_type = "leave_one_out"
        ## DEFINING OUTPUT PATH
        output_path = os.path.join( r"R:\scratch\3d_cnn_project\simulations\190515-cross_validation_test", "cross_valid.txt" ) 
        
    else:
        ## ADDING OPTIONS 
        from optparse import OptionParser # for parsing command-line options
        ## RUNNING COMMAND LINE PROMPTS
        use = "Usage: %prog [options]"
        parser = OptionParser(usage = use)
        
        ## MASS FRACTIONS
        parser.add_option("-m", "--massfrac", dest="mass_frac_data", action="callback", type="string", callback=get_list_args,
                  help="For multiple mass fractions, separate each solute name by comma (no whitespace)", default = ['10', '25', '50', '75'])
        ## SOLVENT NAMES
        parser.add_option("-x", "--solvent", dest="solvent_list", action="callback", type="string", callback=get_list_args,
                  help="For multiple solvents, separate each solute name by comma (no whitespace)", default = [ 'DIO', 'GVL', 'THF' ] )
        ## SOLUTE NAMES
        parser.add_option("-s", "--solute", dest="solute_list", action="callback", type="string", callback=get_list_args,
                  help="For multiple solutes, separate each solute name by comma (no whitespace)", default = None)
        
        
        ## VALIDATION NAME
        parser.add_option('-c', '--cross_validation_name', dest = 'cross_validation_name', 
                          help = 'Cross validation name', default = 'solute', type=str)
        ## VALIDATION TYPE
        parser.add_option('-t', '--cross_validation_type', dest = 'cross_validation_type', 
                          help = 'Cross validation type', default = 'leave_one_out', type=str)
        
        ## OUTPUT PATH
        parser.add_option('-o', '--output_path', dest = 'output_path', help = 'Path to output file', default = None)
        
        ### GETTING ARGUMENTS
        (options, args) = parser.parse_args() # Takes arguments from parser and passes them into "options" and "argument"
        
        ### DEFINING ARGUMENTS
        solute_list = options.solute_list
        solvent_list = options.solvent_list
        mass_frac_data = options.mass_frac_data
        
        ## CROSS VALIDATION TYPES
        cross_validation_name = options.cross_validation_name
        cross_validation_type = options.cross_validation_type
        
        ## DEFINING OUTPUT PATH
        output_path = options.output_path
    
    # -------------- MAIN FUNCTIONS --------------#
    ## DEFINING INPUT VARIALBES
    input_variables = {
            'cosolvent'        : solvent_list,
            'mass_frac'        : mass_frac_data, 
            'solute'           : solute_list,
            }
    
    
    ## DEFINING INPUTS
    cross_validation_inputs = {
            'input_variables'           : input_variables,
            'cross_validation_name'     : cross_validation_name,
            'cross_validation_type'     : cross_validation_type,
            'output_path'               : output_path,
            }
    
    ## RUNNING CROSS VALIDATION SCRIPT
    validation = cross_validation( **cross_validation_inputs )
