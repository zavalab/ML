# -*- coding: utf-8 -*-
"""
import_tools.py
This contains codes on importing functions.

Created by: Alex K. Chew (alexkchew@gmail.com, 02/27/2018)

*** UPDATES ***
20180313 - AKC - Added total residues to trajectory function
20180622 - AKC - Added total frames to trajectory function
20180706 - AKC - Added functions to read file as a line

FUNCTIONS:
    read_file_as_line: reads a file as a line

CLASSES:
    import_traj: class that can import trajectory information (uses md.load from mdtraj module)
    read_gromacs_xvg: class to read gromacs xvg information
        USAGE EXAMPLE: 
            ## READING FILES
            self.output_xvg = import_tools.read_gromacs_xvg(    traj_data = traj_data,
                                                                xvg_file = xvg_file,
                                                                variable_definition = self.variable_definition
                                                            )
        
"""

### TRYING TO LOAD MDTRAJ
#try:
#    # MDTRAJ TO READ TRAJECTORIES
#    import mdtraj as md
#except ModuleNotFoundError:
#    pass

# FUNCTIONS TO CHECK THE PATH OF THE SERVER
from MDDescriptors.core.initialize import checkPath2Server
# FUNCTION TO MEASURE TIME
import time
## MATH MODULE
import numpy as np

## DEFINING VARIABLES
GMX_XVG_VARIABLE_DEFINITION = {'GYRATE': 
                                    [ 
                                    [0, 'frame',    int],
                                    [1, 'Rg',       float ],
                                    [2, 'Rg_X',     float],
                                    [3, 'Rg_Y',     float],
                                    [4, 'Rg_Z',     float],
                                ],
                            'density.xvg':
                                [
                                [0, 'distance', float,], # kg/m^3
                                [1, 'density', float,],  #  kg/m^3
                                ],
                            'potential.xvg':
                                [
                                [0, 'distance', float,], # nm
                                [1, 'potential', float,],  #  Volts
                                ],
                               }


### FUNCTION TO READ FILE AND CONVERT THE INTO LINES
def read_file_as_line(file_path, want_clean = True, verbose = True):
    '''
    The purpose of this function is to read a file and convert them into lines
    INPUTS:
        file_path: [str] full path to your file
        want_clean: [logical, default = True] True if you want to clean the data of '\n'
    OUTPUTS:
        data_full: [list] Your file as a list of strings
    '''
    ## PRINTING
    if verbose is True:
        print("READING FILE FROM: %s"%(file_path))
    ## OPENING FILE AND READING LINES
    with open(file_path, 'r') as file:
        data_full= file.readlines()
    ## CLEANING FILE OF ANY '\n' AT THE END
    if want_clean == True:
        data_full = [s.rstrip() for s in data_full]
    return data_full
    
### FUNCTION TO READ THE XVG FILE
def read_xvg(file_path):
    '''
    The purpose of this function is to read the file and eliminate all comments
    INPUTS:
        file_path: [str] full file path to xvg file
    OUTPUTS:
        self.data_full: [list] full list of the original data
        self.data_extract: [list] extracted data in a form of a list (i.e. no comments)
    '''
    ## PRINTING
    print("READING FILE FROM: %s"%(file_path))
    ## OPENING FILE AND READING LINES
    with open(file_path, 'r') as file:
        data_full= file.readlines()
        
    ## EXTRACTION OF DATA WITH NO COMMENTS (E.G. '@')
    final_index =[i for i, j in enumerate(data_full) if '@' in j][-1]
    data_extract = [ x.split() for x in data_full[final_index+1:] ]
    return data_full, data_extract


####################################
#### CLASS FUNCTION: import_traj ###
####################################
# This class imports all trajectory information
class import_traj:
    '''
    INPUTS:
        directory: directory where your information is located
        self.file_structure: name of your structure file
        xtc_file: name of your xtc file
        want_only_directories: [logical, default = False] If True, this function will no longer load the trajectory. It will simply get the directory information
    OUTPUTS:
        ## FILE STRUCTURE
            self.directory: directory your file is in
            
        ## TRAJECTORY INFORMATION
            self.traj: trajectory from md.traj
            self.topology: toplogy from traj
            self.residues: Total residues as a dictionary
                e.g. {'HOH':35}, 35 water molecules
            self.num_frames: [int] total number of frames
        
    FUNCTIONS:
        load_traj_from_dir: Load trajectory from a directory
        print_traj_general_info: prints the current trajectory information
    '''
    ### INITIALIZING
    def __init__(self, directory, structure_file, xtc_file, want_only_directories = False):
        ### CHECKING THE PATH AND CORRECTING
        directory = checkPath2Server(directory)
        
        ### STORING INFORMATION
        self.directory = directory
        self.file_structure = structure_file
        self.file_xtc = xtc_file
        
        if want_only_directories == False:
            ### START BY LOADING THE DIRECTORY
            self.load_traj_from_dir()
            
            # PRINTING GENERAL TRAJECTORY INFORMATION
            self.print_traj_general_info()
    
    
    ### FUNCTION TO LOAD TRAJECTORIES
    def load_traj_from_dir(self):
        '''
        The purpose of this function is to load a trajectory given an xtc, gro file, and a directory path
        INPUTS:
            self: class object
        OUTPUTS:
            self.traj: [class] trajectory from md.traj
            self.topology: [class] toplogy from traj
            self.num_frames: [int] total number of frames
        '''
        ## CHECKING PATHS
        ## CHECKING IF THE DIRECTORY AS A SLASH AT THE END
        if self.directory[-1] == '/':
            self.path_xtc = self.directory + self.file_xtc
            self.path_structure = self.directory + self.file_structure
        else:
            self.path_xtc = self.directory + '/' + self.file_xtc
            self.path_structure = self.directory + '/' + self.file_structure
    
        ## PRINT LOADING TRAJECTORY
        print('\nLoading trajectories from: %s'%(self.directory))
        print('XTC File: %s' %(self.path_xtc))
        print('Structure File: %s' %(self.path_structure) )
        
        ## LOADING TRAJECTORIES
        start_time = time.time()
        self.traj =  md.load(self.path_xtc, top=self.path_structure)
        print("--- Total Time for MD Load is %s seconds ---" % (time.time() - start_time))
        
        ## GETTING TOPOLOGY
        self.topology=self.traj.topology
        
        ## GETTING TOTAL TIME
        self.num_frames = len(self.traj)
        
        return 
    
    ### FUNCTION TO PRINT GENERAL TRAJECTORY INFORMATION
    def print_traj_general_info(self):
        '''This function simply takes your trajectory and prints the residue names, corresponding number, and time length of your trajectory
        INPUTS:
            self: class object
        OUTPUTS:
            Printed output
        '''
        ## Defining functions ##
        def findUniqueResNames(traj):
            ''' This function simply finds all the residues in your trajectory and outputs its unique residue name
            INPUTS:
                traj: trajectory from md.traj
            OUTPUTS:
                List of unique residues
            '''
            return list(set([ residue.name for residue in traj.topology.residues ]))
    
        def findTotalResidues(traj, resname):
            '''This function takes your residue name and finds the residue indexes and the total number of residues
            INPUTS:
                traj: trajectory from md.traj
                resname: Name of your residue
            OUTPUTS:
                num_residues, index_residues
            '''
            # Finding residue index
            index_residues = [ residue.index for residue in traj.topology.residues if residue.name==resname ]
            
            # Finding total number of residues
            num_residues = len(index_residues)
            
            return num_residues, index_residues
        
        ## Main Script ##
        
        print("---- General Information about your Trajectory -----")
        print("%s\n"%(self.traj))
        
        ## STORING TOTAL RESIDUES
        self.residues={}
        
        # Finding unique residues
        unique_res_names = findUniqueResNames(self.traj)
        for currentResidueName in unique_res_names:
            # Finding total number of residues, and their indexes    
            num_residues, index_residues = findTotalResidues(self.traj, resname = currentResidueName)
            
            ## STORING
            self.residues[currentResidueName] = num_residues
            
            # Printing an output
            print("Total number of residues for %s is: %s"%(currentResidueName, num_residues))
            
        # Finding total time length of simulation
        print("\nTime length of trajectory: %s ps"%(self.traj.time[-1] - self.traj.time[0]))
 
        return
    


########################################################
### DEFINING GENERALIZED XVG READER FOR GMX COMMANDS ###
########################################################
class read_gromacs_xvg:
    '''
    The purpose of this class is to read xvg files in a generalized fashion. Here, you will input the xvg file, 
    define the bounds of the xvg such that the columns are defined. By defining the columns, we will read the xvg file, 
    then extract the information. 
    INPUTS:
        traj_data: [object]
            trajectory data indicating location of the files
        xvg_file: [str]
            name of the xvg file
        variable_definition: [list]
            Here, you will define the variables such that you define the column, name, and type of variable.
            Note: the name of the variable will be used as a dictionary.
    OUTPUTS:
        ## INPUT INFORMATION
            self.variable_definition: [list]
                same as input -- definition of variables
        ## FILE PATHS
            self.file_path: [str]
                full path to the xvg file
        ## FILE INFORMATION
            self.data_full: [list]
                data with full information
            self.data_extract: [list]
                extracted data (no comments)
        ## VARIABLE EXTRACTION
            self.output: [dict]
                output data from defining the variables in a form of a dictionary
    FUNCTIONS:
        define_variables: this function extracts variable details
            
    '''
    ## INITIALIZING
    def __init__(self, traj_data, xvg_file, variable_definition ):
        
        ## STORING INPUTS
        self.variable_definition = variable_definition
        
        ## DEFINING FULL PATH
        self.file_path = traj_data.directory + '/' + xvg_file
        
        ## READING THE FILE
        self.data_full, self.data_extract = read_xvg(self.file_path)

        ## VARIABLE EXTRACTION
        self.define_variables()
    
    ## EXTRACTION OF VARIABLES
    def define_variables(self,):
        '''
        The purpose of this function is to extract variables from column data
        INPUTS:
            self: [object]
                class property
        OUTPUTS:
            self.output: [dict]
                output data from defining the variables in a form of a dictionary
        '''
        ## DEFINING EMPTY DICTIONARY
        self.output={}
        ## LOOPING THROUGH EACH CATEGORY
        for each_variable_definition in self.variable_definition:
            ## DEFINING CURRENT INPUTS
            col = each_variable_definition[0]
            name = each_variable_definition[1]
            var_type = each_variable_definition[2]
            
            ## EXTRACTING AND STORING
            self.output[name] = np.array([ x[col] for x in self.data_extract]).astype(var_type)
        return
        