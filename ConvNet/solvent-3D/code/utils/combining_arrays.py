# -*- coding: utf-8 -*-
"""
combining_arrays.py
The purpose of this script is to loop through the data and generate a pickle 
for the combined array. This is intended to speed up the computation by avoiding 
loading and reloading of multiple pickle files. 

Created on: 04/17/2019

FUNCTIONS/CLASSES:
    renormalize_rgb_array: renormalizes RGB arrays from 0 to 1
    combine_training_data: combines training instances a specific way
    get_combined_name: gets combined name for a representation
    read_combined_name: reverses the get combined name
    combine_instances: class function that actually combines the instances

Author(s):
    - Alex K. Chew (alexkchew@gmail.com)
    - ADD USERS HERE
"""
## IMPORTING NECESSARY MODULES
import os
## IMPORTING PANDAS
import pandas as pd
## IMPORTING NUMPY
import numpy as np
## IMPORTING PICKLE
import pickle

## CHECKING TOOLS
from core.check_tools import check_testing
## IMPORTING GLOBAL VARIABLES
from core.global_vars import SOLUTE_TO_TEMP_DICT
## TAKING EXTRACTION SCRIPTS
from extraction_scripts import load_pickle,load_pickle_general
## TAKING NOMENCLATURE
from core.nomenclature import convert_to_single_name
## IMPORTING ML FUNCTIONS
from core.ml_funcs import locate_test_instance_value, get_list_args, get_split_index_of_list_based_on_percentage
## IMPORTING PATH FUNCTIONS
from core.path import find_paths
## IMPORTING CALC TOOLS
import core.calc_tools_no_mdtraj as calc_tools
## IMPORTING RENORMALIZATION ARRAY
from core.plotting_scripts import renormalize_rgb_array
## IMPORTING FUNCTIONS
from core.nomenclature import get_combined_name, read_combined_name, extract_representation_inputs

### FUNCTION TO CONVERT TRAINING DATA BASED ON A DESIRED REPRESENTATION
def combine_training_data( training_data_for_instance,
                           representation_type,
                           representation_inputs,
                           str_output = None):
    '''
    The purpose of this function is to combine training data into a desired 
    representation. For instance, suppose you want your training data to look 
    a certain way (e.g. averaging across). Note that this requires a valid 
    representation type.
    
    REPRESENTATION TYPES:
        split_avg_withnorm: 
            splits the trajectory, then averages and normalizes RGB array from 0 to 1
        split_avg_nonorm:
            splits the trajectory, then averages (no normalization post averaging)
        split_avg_nonorm_planar:
            same thing as 'split_avg_nonorm', except it averages across the x, y, and z planes to make 2D images
        split_avg_nonorm_perc:
            same as usual, except use a percentage of the trajectories (e.g. 10% means 10% of each trajectory)
    INPUTS:
        training_data_for_instance: [np.array, shape=(num_time, num_voxel, num_voxel, num_voxel, R, G, B)]
            training data instance
        representation_type: [str]
            representation type as a string
        representation_inputs: [dict]
            representation inputs as a dictionary
    OUTPUTS:
        output_training_data: [list]
            output training data
    '''
    ## AVERAGE REPRESENTATION
    if representation_type == 'split_avg_withnorm': 
        ## FINDING NUMBER OF SPLITS
        num_splits = representation_inputs['num_splits']
        ## SPLITTING TRAINING DATA INSTANCE AND AVERAGING
        split_training_instance = calc_tools.split_list( training_data_for_instance, num_splits)
        ## AVERAGING THE DATA
        avg_training_data = [  np.average(split_training_instance[each_split], axis = 0) 
                                    for each_split in range(len(split_training_instance))]
        ## UPDATING BY NORMALIZING
        output_training_data = [ renormalize_rgb_array(each_avg) for each_avg in avg_training_data]
    elif representation_type == 'split_avg_nonorm' \
         or representation_type == 'split_avg_nonorm_planar' \
         or representation_type == 'split_avg_nonorm_50' \
         or representation_type == 'split_avg_nonorm_25' \
         or representation_type == 'split_avg_nonorm_01' \
         or representation_type == 'split_avg_nonorm_perc' \
         or representation_type == 'split_avg_nonorm_sampling_times' \
         :
        
        ## TESTING IF THE TOTAL FRAMES ARE SPECIFIED
        try:
            total_frames = representation_inputs['total_frames']
        except: 
            ## DEFINING TOTAL FRAMES
            total_frames = len(training_data_for_instance)
        
        ## FINDING INDEX OF THE TOTAL FRAMES
        index_first_frame = len(training_data_for_instance) - total_frames
             
        ## FINDING NUMBER OF SPLITS
        num_splits = representation_inputs['num_splits']
    
        ## GETTING NEW TRAINING DATA BASED ON INSTANCES
        training_data_for_instance = training_data_for_instance[index_first_frame:]
        
        
        ## TESTING IF INITIAL AND LAST FRAMES SPECIFIED
        try:
            initial_frame = representation_inputs['initial_frame']
        except:
            ## DEFINING INITIAL FRAME
            initial_frame = 0
        
        ## FINAL FRAME
        try:
            last_frame = representation_inputs['last_frame']
        except:
            last_frame = len(training_data_for_instance)
        
        ## CORRECTING FOR INITIAL AND LAST FRAME
        training_data_for_instance = training_data_for_instance[initial_frame:last_frame]
        
        ## SPLITTING TRAINING DATA INSTANCE AND AVERAGING
        split_training_instance = calc_tools.split_list( training_data_for_instance, num_splits)
        
        ## ADDING TO STRING
        str_output= "%d total frames"%(len(training_data_for_instance))
        
        ## CHECKING IF YOU WANT A SHORTER SPLIT TIME
        if representation_type == 'split_avg_nonorm_50' \
            or representation_type == 'split_avg_nonorm_25' \
            or representation_type == 'split_avg_nonorm_01' \
            or representation_type == 'split_avg_nonorm_perc' \
            or representation_type == 'split_avg_nonorm_sampling_times' \
            :

            ## IF YOU HAVE SMALLER TIME INCREMENTS (50%)
            if representation_type == 'split_avg_nonorm_50':
                ## DEFINING PERCENTAGE
                split_percentage = 0.50
            elif representation_type == 'split_avg_nonorm_25':
                ## DEFINING PERCENTAGE
                split_percentage = 0.25
            elif representation_type == 'split_avg_nonorm_01':
                ## DEFINING PERCENTAGE
                split_percentage = 0.01
            elif representation_type == 'split_avg_nonorm_perc' or representation_type == 'split_avg_nonorm_sampling_times':
                split_percentage = representation_inputs['perc']
        
            ## PRINTING
            str_output = str_output + ", %d split frames, %d percent of the data"%( len(split_training_instance[0]),  int(split_percentage*100) )
            
            ## FINDING SPLITTING 
            split_indexes = [ get_split_index_of_list_based_on_percentage( input_list = current_array,
                                                                         split_percentage = split_percentage,
                                                                        ) for current_array in split_training_instance ]
            
            ## SPLITTING THE DATA
            split_training_instance = [ each_split[:split_indexes[idx]]
                                        for idx, each_split in enumerate(split_training_instance)]
            ## UPDATING SPLITITNG DATA
            str_output = str_output + ", %d output frames"%(len(split_training_instance[0]))
            
            ## UPDATING WITH INITIAL AND LAST FRAMES
            if representation_type == 'split_avg_nonorm_sampling_times':
                str_output = str_output + ", %d frame - %d frame"%(initial_frame, last_frame )
            

        ## AVERAGING THE DATA
        avg_training_data = [  np.average(split_training_instance[each_split], axis = 0) 
                                    for each_split in range(len(split_training_instance))]
        

        ## IF YOU HAVE A PLANAR SURFACE
        if representation_type == 'split_avg_nonorm_planar':
            ## CREATING EMPTY ARRAY
            output_training_data = []
            ## LOOPING THROUGH EACH OUTPUT TRAINING DATA
            for each_avg_training in avg_training_data:
                ## FINDING AVERAGE FOR X, Y, AND Z DIMENSIONS
                x_avg = np.mean(each_avg_training, axis=0)
                y_avg = np.mean(each_avg_training, axis=1)
                z_avg = np.mean(each_avg_training, axis=2)
                ## STORING
                output_training_data.extend([x_avg, y_avg, z_avg])
        else:
            output_training_data = avg_training_data
        
    else:
        print("Error! Data representation '%s' not found!"%( representation_type ))
    return output_training_data, str_output
    
###########################################
### CLASS FUNCTION TO COMBINE INSTANCES ###
###########################################
class combine_instances:
    '''
    The purpose of this class is to combine class instances so that we do not 
    have to constantly load multiple pickles. Once the pickle is created, we no 
    longer need to re-create it (that's pretty neat!)
    INPUTS:
        solute_list: [list]
            list of solutes you are interested in
        solvent_list: [list]
            list of solvent data, e.g. [ 'DIO', 'GVL', 'THF' ]
        mass_frac_data: [list]
            list of mass fraction data, e.g. ['10', '25', '50', '75']
        representation_type: [str]
            string of representation types
        representation_inputs: [dict]
            dictionary for the representation input
        enable_pickle: [logical, default=True]
            True if you want to enable pickle loading and unloading. 
            By default, this should speed up computations. Just check your 
            paths!
        verbose: [logical, default=False]
            True if you want to print functions
        data_type: [str, default="20_20_20"]
            data type that you are interested in
        ## PATHS
            NOTE: if any of these are none, we will find paths and correct accordingly!
            database_path: [str, default = None]
                path to the database
            class_file_path: [str, default = None]
                path to the class file spreadsheet
            combined_database_path: [str, default = None]
                path to combined databases
    OUTPUTS:
        ## STORED INPUTS
        self.solvent_list, self.mass_frac_data, self.solute_list, self.representation_type, 
        self.solute_list, self.representation_type, self.representation_inputs, self.verbose
        
        ## PICKLE FILE NAME
        self.pickle_name: [str]
            pickle file name that you will save / reopen in
        
        ## PATHS
        self.database_path: [str]
            path to database
        self.class_file_path: [str]
            path to class file path
        self.combined_database_path: [str]
            combined database path
        self.combined_database_path_pickle: [str]
            path directly to pickle file name
    '''
    ## INITIALIZING
    def __init__(self, 
                 solute_list,
                 representation_type = 'split_average',
                 representation_inputs = { 'num_splits': 5 },
                 solvent_list = [ 'DIO', 'GVL', 'THF' ], 
                 mass_frac_data = ['10', '25', '50', '75'], 
                 data_type = "20_20_20",
                 enable_pickle= True,
                 verbose = False,
                 database_path = None,
                 class_file_path = None,
                 combined_database_path = None,
                 ):
        ## STORING INITIAL VARIABLES
        self.solute_list = solute_list
        self.solvent_list = solvent_list
        self.mass_frac_data = mass_frac_data
        self.solute_list = solute_list
        self.representation_type = representation_type
        self.representation_inputs = representation_inputs
        self.verbose = verbose
        self.database_path = database_path
        self.class_file_path = class_file_path
        self.combined_database_path = combined_database_path
        self.enable_pickle = enable_pickle
        self.data_type = data_type
        
        ## FINDING NAME OF FILE
        self.pickle_name = get_combined_name(
                                             representation_type = self.representation_type,
                                             representation_inputs = self.representation_inputs,
                                             solute_list = self.solute_list,
                                             solvent_list = self.solvent_list,
                                             mass_frac_data = self.mass_frac_data,
                                             data_type = self.data_type
                                             )
        ## CAN REVERSE TO GET REPRESENTATION: combined_name_info = read_combined_name(unique_name)
        
        ## LOCATING ALL PATHS
        self.find_all_path()
        
        ## SEEING IF PICKLE FILE EXISTS
        self.pickle_file_exist = os.path.isfile(self.combined_database_path_pickle)
        
        ## SEEING IF PICKLE EXISTS
        if self.pickle_file_exist == True and self.enable_pickle == True:
            print("---------------------------------------------------------------")
            print("Pickle found! Loading the following pickle:")
            print(self.combined_database_path_pickle)
            print("---------------------------------------------------------------")
            ## LOADING THE PICKLE
            self.restore_pickle()
        else:
            ## PRINTING
            print("---------------------------------------------------------------")
            print("Pickle was not found! If this is an error, please check path: ")
            print(self.combined_database_path_pickle)
            print("---------------------------------------------------------------")
            ## LOADING ALL XY DATA
            self.load_xy_data()
            ## RESAVING THE PICKLE
            self.store_pickle()
        
        return
        
    ### FUNCTION TO STORE PICKLE
    def store_pickle(self):
        ''' This function stores the pickle'''
        ## CHECKING IF DIRECTORY EXISTS
        if os.path.isdir( self.combined_database_path ) == False:
            print("Creating directory in %s"%(self.combined_database_path))
        
        with open(self.combined_database_path_pickle, 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump([self.x_data, self.y_label, self.instance_names], f, protocol=2)  # <-- protocol 2 required for python2   # -1
        # print("Data collection was complete at: %s\n"%(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        
    ### RETRIVE PICKLE
    def restore_pickle(self):
        ''' This function restores the pickle '''
        self.x_data, self.y_label, self.instance_names = load_pickle_general(self.combined_database_path_pickle)
            
    ### FUNCTION TO FIND PATHS
    def find_all_path(self):
        ''' This function looks for all paths to databases, etc. '''
        ## FINDING PATHS
        path_dict = find_paths()
        ## DEFINING PATH TO DATABASE
        if self.database_path == None:
            self.database_path = path_dict['database_path']
        if self.class_file_path == None:
            ## DEFINING PATH TO CLASS FILE
            self.class_file_path = path_dict['csv_path']
        if self.combined_database_path == None:
            ## DEFINING OUTPUT FILE PATH
            self.combined_database_path = path_dict['combined_database_path']
        ## DEFINING COMBINED PATH PICKLE
        self.combined_database_path_pickle = os.path.join( self.combined_database_path, self.pickle_name )
        
        return
        
    ### FUNCTION TO LOAD X Y DATA
    def load_xy_data(self):
        '''
        This function loads all desired xy data. 
        '''    
        ## READING CSV FILE
        csv_file = pd.read_csv( self.class_file_path )
        
        ## STORE X DATA AND ITS LABEL (POSITIVE OR NEGATIVE)
        self.x_data = []
        self.y_label = []
        
        ## STORING INSTANCE INFORMATION
        self.instance_names = []
        ## LOOPING THROUGH SOLUTES
        for solute in self.solute_list:
            ## LOOPING THROUGH COSOLVENT
            for cosolvent in self.solvent_list:
                ## LOOPING THROUGH MASS FRACTION OF WATER
                for mass_frac in self.mass_frac_data:
                    ## SPECIFYING SPECIFIC TRAINING INSTANCE
                    training_instance = {
                            'solute': solute,
                            'cosolvent': cosolvent,
                            'mass_frac': mass_frac, # mass fraction of water
                            'temp': SOLUTE_TO_TEMP_DICT[solute], # Temperature
                            }
                    
                    ## CONVERTING TRAINING INSTANCE NAME TO NOMENCLATURE
                    training_instance_name = convert_to_single_name( 
                                                                    solute = training_instance['solute'],
                                                                    solvent = training_instance['cosolvent'],
                                                                    mass_fraction = training_instance['mass_frac'],
                                                                    temp = training_instance['temp']
                                                                    )
                    
                    ##############################
                    ### EXTRACTING CLASS VALUE ###
                    ##############################
                    ## FINDING INSTANCE VALUE
                    class_instance_value = locate_test_instance_value(
                                                                        csv_file = csv_file,
                                                                        solute =  training_instance['solute'],
                                                                        cosolvent = training_instance['cosolvent'],
                                                                        mass_frac_water = training_instance['mass_frac'],
                                                                        )
                    ## PRINTING

                    if (str(class_instance_value) != 'nan'):
                        self.y_label.append(class_instance_value)
                    else:
                        print("Training instance: %s, Class value: %s -- skipping!"%( training_instance_name, class_instance_value ) )
                
                    ################################
                    ### EXTRACTING TRAINING SETS ###
                    ################################
                    if (str(class_instance_value) != 'nan'):

                        ## DEFINING FULL TRAINING PATH
                        full_train_pickle_path= os.path.join(self.database_path, training_instance_name)
                        ## PRINTING
                        print(full_train_pickle_path)
                        ## EXTRACTION PROTOCOL A PARTICULAR TRAINING EXAMPLE
                        training_data_for_instance = load_pickle(full_train_pickle_path)
                        ## CHANGING TRAINING DATA INSTANCE REPRESENTATION
                        training_data_representation, str_output = combine_training_data( training_data_for_instance = training_data_for_instance,
                                                                              representation_type = self.representation_type,
                                                                              representation_inputs = self.representation_inputs)
                        
                        ## PRINTING
                        if self.verbose == True:
                            if str_output is None:
                                print("Instance: %s, Class value: %s"%( training_instance_name, class_instance_value ) ) # Should output negative
                            else:
                                print("Instance: %s, Class value: %s, %s"%( training_instance_name, class_instance_value, str_output ) ) # Should output negative
                        
                        ## STORING
                        self.x_data.append(training_data_representation)
                        self.instance_names.append(training_instance_name)
        ## FINDING TOTAL INSTANCES
        self.total_instances = len(self.instance_names)
        return
    ### FUNCTION TO PLOT
    def plot_voxel_instance(self, instance_index = 0, frame = 0,  want_renormalize = False, want_split = False ):
        '''
        The purpose of this function is to plot a voxel using the data that is available. 
        We are assuming that you have already extracted the training data.
        INPUTS:
            self: [obj]
                class object
            frame: [int]
                frame you are interested in
            want_split: [logical, default=False]
                True if you want to split plot in half.
            instance_index: [int, default = 0]
                instance index you are interested in. Please use self.instance_names to designate which instance you want.
        OUTPUTS:
            fig, ax: [obj]
                figure object and axis
        '''
        ## PLOTTING MODULES
        from core.plotting_scripts import plot_voxel
        ## IF SPLIT AVERAGE, PRINT THE FIRST AVERAGE
        if self.representation_type == 'split_average' or self.representation_type == 'split_avg_nonorm':
            fig, ax = plot_voxel(grid_rgb_data = self.x_data[instance_index][frame], frame = frame, want_split = want_split, want_renormalize = want_renormalize,)
        else:
            fig, ax = plot_voxel(grid_rgb_data = self.x_data[instance_index], frame = frame, want_split = want_split, want_renormalize = want_renormalize,)
        return fig, ax

#%%
## MAIN FUNCTION
if __name__ == "__main__":
    
    ## SEE IF TESTING IS ON
    testing = check_testing()
    
    if testing == True:
    
        ## DEFINING SOLVENT LIST
        solvent_list = [ 'DIO', 'GVL', 'THF' ]#  ]
        ## DEFINING MASS FRACTION DATA
        mass_frac_data = ['10', '25', '50', '75'] # ,   '75' ,'25', '50',
        ## DEFINING SOLUTE LIST
        solute_list = list(SOLUTE_TO_TEMP_DICT) #  ['XYL'] #
        ## DEFINING TYPE OF REPRESENTATION
        representation_type = 'split_avg_nonorm' 
        # split_avg_withnorm
        # split_avg_nonorm
        # split_avg_nonorm_planar -- projection onto a 2D plane
        # split_avg_nonorm_50 -- avg norm with 50% of the data
        # split_avg_nonorm_25 -- avg norm with 25% of the data
        # split_avg_nonorm_01 -- avg norm with 1% of the data
        # split_avg_nonorm_perc -- avg norm with any % data
        # split_avg_nonorm_sampling_times -- same as above but with initial and last frames indicated
        # FOR split_avg_nonorm_perc:
        
        representation_inputs = {
                'num_splits': 8,
                }
        
#        representation_inputs = {
#                'num_splits': 5,
#                'perc': 0.05,
#                'total_frames': 5000,
#                }
#        representation_inputs = {
#                'num_splits': 5,
#                'perc': 1.00,
#                'initial_frame': 15000,
#                'last_frame': 19000,
                # 
        ## DEFINING DATA TYPE
        data_type="20_20_20_40ns_first"  # 20_20_20_rdf
        # 20_20_20_190ns
        # 20_20_20_withdmso
        # 20_20_20
        # 30_30_x30
        # 32_32_32
        # 20_20_20_rdf -- with RDF
        # 20_20_20_withdmso -- with DMSO
        
#        ## DEFINING PATH TO HOME
#        home_path = r"R:\scratch"
#        home_path = r"/Volumes/akchew/scratch"
#        home_path = r"/home/akchew/scratch"
#        
#        ## DEFINING PATHS
#        database_path = os.path.join( home_path, r"3d_cnn_project/database", data_type) # None # Since None, we will find them!
#        class_file_path = os.path.join( home_path, r"3d_cnn_project/database/Experimental_Data/solvent_effects_regression_data.csv")  # None
#        output_file_path = os.path.join( home_path, r"3d_cnn_project/simulations" ) # OUTPUT PATH FOR CNN NETWORKS
#        combined_database_path = os.path.join( home_path, r"3d_cnn_project/combined_data_set"  )       
        
        ## DEFINING PATHS
        database_path = os.path.join( r"R:\scratch\3d_cnn_project\database", data_type)  # None # Since None, we will find them!        
        class_file_path = None
        combined_database_path = r"R:\scratch\3d_cnn_project\combined_data_set"
        output_file_path = r"R:\scratch\3d_cnn_project\simulations" # OUTPUT PATH FOR CNN NETWORKS
        
        ## DEFINING VERBOSITY
        verbose = True
        
        ## TURNING OFF PICKLEING
        enable_pickle = True # false
        
    else:
        from optparse import OptionParser # for parsing command-line options
        ## RUNNING COMMAND LINE PROMPTS
        use = "Usage: %prog [options]"
        parser = OptionParser(usage = use)

        ## REPRESENTATION TYPE
        parser.add_option('-r', '--representation', dest = 'representation_type', help = 'Representation type', default = 'split_average', type=str)
        parser.add_option("-g", "--representation_inputs", dest="representation_inputs", action="callback", type="string", callback=get_list_args,
                  help="For multiple inputs, simply separate by comma (no whitespace)", default = 5)
        ## MASS FRACTIONS
        parser.add_option("-m", "--massfrac", dest="mass_frac_data", action="callback", type="string", callback=get_list_args,
                  help="For multiple mass fractions, separate each solute name by comma (no whitespace)", default = ['10', '25', '50', '75'])
        ## SOLVENT NAMES
        parser.add_option("-x", "--solvent", dest="solvent_list", action="callback", type="string", callback=get_list_args,
                  help="For multiple solvents, separate each solute name by comma (no whitespace)", default = [ 'DIO', 'GVL', 'THF' ] )
        ## SOLUTE NAMES
        parser.add_option("-s", "--solute", dest="solute_list", action="callback", type="string", callback=get_list_args,
                  help="For multiple solutes, separate each solute name by comma (no whitespace)", default = None)
        
        ## DEFINING DATA SET TYPE
        parser.add_option('-z', '--datatype', dest = 'data_type', help = 'data type', type="string", default = "20_20_20")
        
        ## DIRECTORY LOCATIONS
        parser.add_option('-d', '--database', dest = 'database_path', help = 'Full path to database', default = None)
        parser.add_option('-c', '--classfile', dest = 'class_file_path', help = 'Full path to class csv file', default = None)
        parser.add_option('-a', '--combinedfile', dest = 'combined_database_path', help = 'Full path to combined pickle directory', default = None)
        parser.add_option("-v", dest="verbose", action="store_true", )
        parser.add_option("-p", dest="enable_pickle", action="store_false", )
        
        ### GETTING ARGUMENTS
        (options, args) = parser.parse_args() # Takes arguments from parser and passes them into "options" and "argument"
        
        ### DEFINING ARUGMENT
        solute_list = options.solute_list
        solvent_list = options.solvent_list
        mass_frac_data = options.mass_frac_data
        ## REPRESENTATION
        representation_type = options.representation_type
        representation_inputs = options.representation_inputs
        ## FILE PATHS
        database_path = options.database_path
        class_file_path = options.class_file_path
        combined_database_path = options.combined_database_path
        ## DATA  SET TYPE
        data_type = options.data_type
        # 20_20_20_40ns_first-split_avg_nonorm-8-CEL_ETBE_FRU_LGA_PDO_XYL_tBuOH-DIO_GVL_THF-10_25_50_75
        
        ## PICKLE
        enable_pickle = options.enable_pickle
        
        ## VERBOSITY
        verbose = options.verbose
        
        ## UPDATING REPRESATION INPUTS
        representation_inputs = extract_representation_inputs( representation_type = representation_type, 
                                                               representation_inputs = representation_inputs )
        
        
    ## CORRECTING FOR SOLUTE LIST IF NOT AVAILABLE
    if solute_list == None:
        solute_list = list(SOLUTE_TO_TEMP_DICT)
        

    ## RUNNING COMBINED INSTANCES
    instances = combine_instances(
                     solute_list = solute_list,
                     representation_type = representation_type,
                     representation_inputs = representation_inputs,
                     solvent_list = solvent_list, 
                     mass_frac_data = mass_frac_data, 
                     verbose = verbose,
                     database_path = database_path,
                     class_file_path = class_file_path,
                     combined_database_path = combined_database_path,
                     data_type = data_type,
                     enable_pickle = enable_pickle, # True if you want pickle on
                     )

    #%%
    
    ## PLOTTING SPECIFIC INSTANCE
    # instances.plot_voxel_instance()
    
    
    #%%
    
    
#    ## DEFINING PICKLE PATH
#    full_train_pickle_path = r"/Volumes/akchew/scratch/3d_cnn_project/database/20_20_20_190ns/XYL_403.15_DIO_10"
#
#    training_data_for_instance = load_pickle(full_train_pickle_path)
#    
#    #%%
#    ## DEFINING ANALYSIS
#    with open(full_train_pickle_path, 'rb') as pickle_file:
#        training_data_for_instance = pickle.load(pickle_file, encoding = 'latin1') # , encoding = 'latin1'
#        # training_data_for_instance = pd.read_pickle( pickle_file  )
#    
    # 
    

    #%%
#    
#    # --- VISUALIZING 3D IMAGE
#    ## IMPORTING MODULES
#    import matplotlib.pyplot as plt
#    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
#    import numpy as np
#    from core.plotting_scripts import plot_voxel, renormalize_rgb_array
#    
#    instance_index = instances.instance_names.index('CEL_403.15_DIO_10') # 'XYL_403.15_DIO_10'
#    
#    ## FRAME
#    frame = 0
#    
#    fig, ax = plot_voxel(grid_rgb_data = instances.x_data[instance_index][frame], 
#                         frame = frame, 
#                         want_split = False, 
#                         want_renormalize = True,)
#    
#    
#    #%%
#    
#    ## IMPORT SCIPY
#    import scipy.ndimage as ndimage
#    
#    ## DEFINING INSTANCE INDEX
#    instance_index = instances.instance_names.index('CEL_403.15_DIO_10') # 'XYL_403.15_DIO_10'
#    
#    ## DEFINING X DATA
#    x_data =  np.array(instances.x_data[instance_index])
#    ## ROTATING
#    x_train_xy_1 = ndimage.interpolation.rotate(x_data, 0, (1,2))
#    
#    ## FRAME
#    frame = 0
#    
#    # grid_rgb_data = x_data[frame]
#    grid_rgb_data = x_train_xy_1 # [frame]
##    grid_rgb_data = x_data[frame]
#    
#
#    fig, ax = plot_voxel(grid_rgb_data = grid_rgb_data, 
#                         frame = frame, 
#                         want_split = False, 
#                         want_renormalize = True,)
#    ## CHANGING AXIS
#    # ax.view_init(azim=0, elev=90) # XY PLANE
#    
#    # ax.view_init(azim=0, elev=0) # ZY PLANE
#    
#    # ax.view_init(azim=90, elev=0) # XZ PLANE
#    
#    #%%
#    instances.plot_voxel_instance(instance_index = instance_index,
#                                  frame = 1, 
#                                  want_renormalize = False)
    
    
    
    
    #%%
    # --- VISUALIZING 2D IMAGE
    
#    ## AVERAGING ALONG THE X-DIRECTION
#    instances_x_avged = np.mean(instances.x_data[0][0],axis=2)
#    
#    #--- VISUALIZING
#    ## IMPORTING MODULES
#    import matplotlib.pyplot as plt
#    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
#    import numpy as np
#    from core.plotting_scripts import renormalize_rgb_array
#    
#    instances_x_avged_norm =  renormalize_rgb_array(instances_x_avged) # instances_x_avged #
#    instances_x_avged_norm =  instances_x_avged
#    
#    ## FINDING SHAPE
#    grid_shape = instances_x_avged.shape
#    
#    ## DEFINING INDICES, E.G. 1 TO 20
#    x, y= np.indices(np.array(instances_x_avged[...,0].shape) + 1)# +1
#    z = np.zeros(x.shape)
#    
#    ## DEFINING VOXELS
#    voxels = (instances_x_avged[...,0] > 0) | \
#             (instances_x_avged[...,1] > 0) | \
#             (instances_x_avged[...,2] > 0)
#    
#    ## DEFINING COLORS
#    colors = instances_x_avged
#    
#    ## PLOTTING
#    fig, ax = plt.subplots()
#    # fig = plt.figure()
#    # ax = fig.gca(projection='3d')
#    
#    # ax.imshow(X = instances_x_avged_norm, alpha = 0.5, vmax = abs(instances_x_avged_norm).max(), vmin=abs(instances_x_avged_norm).min())
#    ax.imshow(X = instances_x_avged_norm, interpolation='bilinear', alpha = 1)
#    # ax.imshow(X = instances_x_avged_norm, alpha = 1)
    
    #%%

    
    #%%
    ''' Example where you are plotting voxels
    from core.plotting_scripts import plot_voxel

    
    training_instance = {
            'solute': 'XYL',
            'cosolvent': 'DIO',
            'mass_frac': '10', # mass fraction of water
            'temp': SOLUTE_TO_TEMP_DICT['XYL'], # Temperature
            }
    
    ## CONVERTING TRAINING INSTANCE NAME TO NOMENCLATURE
    training_instance_name = convert_to_single_name( 
                                                    solute = training_instance['solute'],
                                                    solvent = training_instance['cosolvent'],
                                                    mass_fraction = training_instance['mass_frac'],
                                                    temp = training_instance['temp']
                                                    )
    
    ## DEFINING FULL TRAINING PATH
    full_train_pickle_path= os.path.join(instances.database_path, training_instance_name)
    ## EXTRACTION PROTOCOL A PARTICULAR TRAINING EXAMPLE
    training_data_for_instance = load_pickle(full_train_pickle_path)
    
    # grid_rgb_data = data
    # r, g, b = np.indices(np.array(grid_rgb_data[0][...,0].shape)+1)
    
    ## IMPORTING PRINT VOXEL FUNCTION
    plot_voxel(grid_rgb_data = training_data_for_instance, frame = 0)
    representation_inputs = {'num_splits': 5}
    ## VIEWING HOW ONE OF THEM MIGHT LOOK LIKE
    full_data = combine_training_data( training_data_for_instance = training_data_for_instance,
                                  representation_type = representation_type,
                                  representation_inputs = representation_inputs)
    
    data = full_data[0]

    plot_voxel(grid_rgb_data = data, frame = 0, want_split = True)
    
    '''