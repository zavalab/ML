#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
loop_grid_interpolation.py
This includes code about grid interpolation
"""
import os
import MDDescriptors.core.import_tools as import_tools # Loading trajectory details
from MDDescriptors.core.initialize import checkPath2Server
from core.check_tools import check_testing, check_path_to_server

## IMPORTING IMPORTANT MODULES
from generate_grid_interpolation import generate_grid_interpolation

## TAKING NOMENCLATURE
from core.nomenclature import convert_to_single_name

## IMPORTING GLOBAL VARIABLES
from core.global_vars import SOLUTE_TO_TEMP_DICT, POSSIBLE_SOLVENTS

## IMPORTING CORE FUNCTIONS
from core.ml_funcs import get_list_args

### FUNCTION TO MAKE DIRECTORY
def mkdir( mydir ):
    ''' This function makes directory, checks before making '''
    try:
       if not os.path.exists(os.path.dirname(mydir)):
           os.makedirs(os.path.dirname(mydir))
    except OSError as err:
       print(err)
    return 


#%%
if __name__ == "__main__":    
    
    ## True if you are debugging on spyder
    testing = check_testing()
    
    ## CHECKING IF YOU ARE TESTING
    if testing is True:
        ## DEFINING INPUT LOCATION
        input_trajectory_location = check_path_to_server(r"R:\scratch\SideProjectHuber\Analysis\\") # PC Side  + analysis_dir + '\\' + specific_dir
        ## DEFINING GRO AND XTC FILE
        gro_file=r"mixed_solv_prod.gro" # Structural file <-- must be a pdb file!
        xtc_file=r"mixed_solv_prod_last_10_ns_centered.xtc"
        ## DEFINING MAPPING  TYPE
        map_type="allatom"
        map_box_size = 4.0 # nm, box size
        map_box_increment = 0.2 # nm box increment
        ## DEFINING OUTPUT DATABASE LOCATION
        output_database_location = os.path.join(checkPath2Server(r"R:\scratch\SideProjectHuber\Analysis\\"), 'cs760_database', map_type )
        ## DEFINING NORMALIZATION
        normalization = 'rdf'
        
        ## DEFINING LIST OF SOLUTES
        solute_list = [ 'ETBE', 'tBuOH', 'PDO', 'LGA', 'FRU', 'CEL', 'XYL' ]
    #    ## DEFINING TEMPERATURE
    #    list_of_temp = [ '343.15', '363.15', '433.15', '403.15', '373.15', '403.15', '403.15']
        ## LIST OF SOLVENTS
        solvent_list = ['DIO', 'GVL', 'THF', 'dmso' ] #  'DIO' 
        ## DEFINING MASS FRACTION DATA
        mass_frac_data = [ '10', '25', '50', '75']
        
    ## TESTING IS FALSE
    else:
        from optparse import OptionParser # for parsing command-line options
        ## RUNNING COMMAND LINE PROMPTS
        parser = OptionParser()
        
        ## PATH INFORMATION
        parser.add_option('-i', '--inputtraj', dest = 'input_trajectory_location', help = 'Input path to trajectories', default = '.')
        parser.add_option('-o', '--outputpath', dest = 'output_database_location', help = 'Output path to trajectories', default = '.')
        
        ## FILE INFORMATION
        parser.add_option('-g', '--grofile', dest = 'gro_file', help = 'Gro file', default = '.')
        parser.add_option('--xtcfile', dest = 'xtc_file', help = 'Xtc file', default = '.')
        
        ## BOX SIZE INFORMATION
        parser.add_option('-b', '--boxsize', dest = 'map_box_size', help = 'Map box size', default = 4.0, type=float)
        parser.add_option('-c', '--boxinc', dest = 'map_box_increment', help = 'Map box increment', default = 0.2, type=float)
        
        ## DEFINING MAPPING TYPE
        parser.add_option('-t', '--maptype', dest = 'map_type', help = 'Mapping type', default = 'allatom')
        
        ## DEFINING NORMALIZATION TYPE
        parser.add_option('-n', '--norm', dest = 'normalization', help = 'Normalization type', default = 'maxima')
        
        ## MASS FRACTIONS
        parser.add_option("-m", "--massfrac", dest="mass_frac_data", action="callback", type="string", callback=get_list_args,
                  help="For multiple mass fractions, separate each solute name by comma (no whitespace)", default = ['10', '25', '50', '75'])
        ## SOLVENT NAMES
        parser.add_option("--solvent", dest="solvent_list", action="callback", type="string", callback=get_list_args,
                  help="For multiple solvents, separate each solute name by comma (no whitespace)", default = [ 'DIO', 'GVL', 'THF' ] )
        ## SOLUTE NAMES
        parser.add_option("-s", "--solute", dest="solute_list", action="callback", type="string", callback=get_list_args,
                  help="For multiple solutes, separate each solute name by comma (no whitespace)", default = None)
        ## TEMPERATURES
        parser.add_option("--temp", dest="temp", action="callback", type="string", callback=get_list_args,
                  help="For multiple solutes, separate each solute name by comma (no whitespace)", default = None)
        ## DESIRED FULL PATH NAMES
        parser.add_option("--fullpath", dest="want_full_path", action="store_true", default = False )
        ## SKIP COSOLVENT NAMING
        parser.add_option("--skipcosnaming", dest="skipcosnaming", action="store_true", default = False )
        ## SPECIFY SOLVENTS
        parser.add_option("--specifysolvents", dest="specifysolvents", action="store_true", default = False )
        ## DESIRED PICKLE NAME
        parser.add_option('--picklesuffix', dest = 'picklesuffix', help = 'Pickle suffix', default = '')	
        
        ### GETTING ARGUMENTS
        (options, args) = parser.parse_args() # Takes arguments from parser and passes them into "options" and "argument"
        
        ### DEFINING ARUGMENTS
        input_trajectory_location = options.input_trajectory_location
        output_database_location = options.output_database_location
        gro_file = options.gro_file
        xtc_file = options.xtc_file
        map_box_size = options.map_box_size
        map_box_increment = options.map_box_increment
        map_type = options.map_type
        normalization = options.normalization
        
        ### DEFINING ARUGMENT
        solute_list = options.solute_list
        solvent_list = options.solvent_list
        mass_frac_data = options.mass_frac_data
        temp = options.temp

        ## DEFINING FULL PATH
        want_full_path = options.want_full_path
        ## COSOLVENT NAMING
        skipcosnaming = options.skipcosnaming
        ## SPECIFYING SOLVENTS
        specifysolvents = options.specifysolvents
        
        ## DEFINING PICKLE SUFFIX
        picklesuffix = options.picklesuffix

    #############################################
    #### RUNNING MAIN SCRIPT 
    #############################################
    
    ## DEFINING NAMES 
    cosolvent_res_to_name = {
            'DIO': 'dioxane',
            'THF': 'tetrahydrofuran',
            'GVL': 'GVL_L',
            'dmso': 'dmso',
            'ACE' : 'acetone',
            'ACN': 'acetonitrile',
            }
    
    ## CREATING DIRECTORY
    mkdir(output_database_location)
    
    ## DEFINING TEMPERATURE LIST
    if temp is None:
        temperature_dict = SOLUTE_TO_TEMP_DICT
    else:
        temperature_dict = { each_solute: temp[idx] for idx, each_solute in enumerate(solute_list) } 
    
    ## LOOPING THROUGH EACH SOLUTE AND SOLVENT
    for idx_solute,solute in enumerate(solute_list):
        ## DEFINING CURRENT TEMPERATURE
        temperature = temperature_dict[solute]
        ## LOOPING THROUGH SOLVENTS
        for solvents in solvent_list:
            ## LOOP THROUGH MASS FRACTION
            for mass_frac in mass_frac_data:
                ## DEFINING COSOLVENT NAME
                if skipcosnaming is False:
                    cosolvent_name = cosolvent_res_to_name[solvents]
                else:
                    cosolvent_name = solvents
                
                ## DEFINING NAME
                directory_name = 'mdRun_' + temperature +   '_6_nm_' + \
                                    solute + '_' + mass_frac + '_WtPercWater_spce_' + cosolvent_name
                                    
                ## DEFINING INPUT LOCATION
                if want_full_path is False:
                    input_dir_path = os.path.join( input_trajectory_location, solute, directory_name)
                else:
                    input_dir_path = os.path.join( input_trajectory_location, directory_name)
                print("Input dir path: %s"%(input_dir_path))
                
                ## CHECKING IF DIRECTORY PATH EXISTS
                if os.path.exists(input_dir_path) is True:
                
                    ## DEFINING NAME FOR PICKLE
                    pickle_name = convert_to_single_name( solute = solute, 
                                                          solvent = solvents, 
                                                          mass_fraction = mass_frac, 
                                                          temp = temperature)

                    ## DEFINING PICKLE PATH
                    pickle_file_path = os.path.join(output_database_location, pickle_name + picklesuffix)

                    ### LOADING TRAJECTORY
                    traj_data = import_tools.import_traj( directory = input_dir_path, # Directory to analysis
                                                          structure_file = gro_file, # structure file
                                                          xtc_file = xtc_file, # trajectories
                                                          )

                    ## RUNNING GRID INTERPOLATION
                    grid_interp_input_vars={
                            'traj_data'         : traj_data,
                            'solute_name'       : solute, # Solute of interest tBuOH
                            'solvent_name'      : POSSIBLE_SOLVENTS , # Solvent of interest HOH 'HOH' , 'GVLL'
                            'map_box_size'      : map_box_size, # nm box length in all three dimensions
                            'map_box_increment' : map_box_increment, # box cell increments
                            'map_type'          : map_type, # mapping type: 'COM' or 'allatom'
                            'normalization'     : normalization, # normalization
                            }
                    ## SEEING IF SOLVENTS WERE SPECIFIED
                    if specifysolvents is True:
                        import core.itp_file_tools as itp_file_tools
                        ## IMPORTING ITP FILE
                        itp_file_path=os.path.join(input_dir_path, solvents + '.itp')
                        # itp_file = read_write_tools.extract_itp(itp_file_path)
                        ## GETTING SOLVENT RESIDUE NAME
                        solvent_residue_name = itp_file_tools.read_residue_name_from_itp(itp_file_path)
                        grid_interp_input_vars['solvent_name'] = ['HOH', solvent_residue_name]

                    ## RUNNING GRID INTERPOLATION
                    grid_interpolation = generate_grid_interpolation( **grid_interp_input_vars )

                    ## STORING TO PICKLE
                    grid_interpolation.store_pickle(pickle_file_path)
                else:
                    print("%s does not exist.. skipping!"%(input_dir_path))
                