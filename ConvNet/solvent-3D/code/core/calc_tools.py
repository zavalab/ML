# -*- coding: utf-8 -*-
"""
calc_tools.py
In this script, we have functions that can operate across trajectories. General functions are listed below:
    ## TRAJECTORY TOOLS
        find_total_residues: Finds the total number of residues and the corresponding indexes to them
        find_atom_index: Finds the atom index based on residue name and atom name
        find_atom_names: Finds atom names given the residue name
        find_specific_atom_index_from_residue: finds atom index from residue name
        find_residue_atom_index: Outputs residue and atom index for a residue of interest
        find_multiple_residue_index: finds multiple residues given a list -- outputs index and total residues
        find_center_of_mass: Calculates the center of mass of residue
        calc_ensemble_vol: calculates ensemble volume given the trajectory
        create_atom_pairs_list: creates atom pair list between two atom lists (very quick!)
        create_atom_pairs_with_self: creates atom pair list for a single atom list (you are interested in atom-atom interactions with itself)
        find_water_index: finds all water index (atom indices)
        calc_pair_distances_with_self_single_frame: calculates pair distances for a single frame
        calc_pair_distances_between_two_atom_index_list: calculates pair distances given two list of atom indices
        
    ## SPLITTING TRAJECTORY FUNCTIONS
        split_traj_function: splits the trajectory and calculates a value. This works well if you are receiving memory errors
        split_traj_for_avg_std: splits the trajectory so you can calculate an average and standard deviation
        calc_avg_std: calculates average and std of a list of dictionaries
        calc_avg_std_of_list: calculates average and standard deviation of a list
        split_list: splits list
        split_general_functions: splits calculations based on generalized inputs *** useful for cutting trajectories and computing X.
        
    ## VECTOR ALGEBRA
        unit_vector: converts vectors to unit vectors
        angle_between: finds the angles between any two vectors in radians
        rescale_vector: rescales vectors and arrays from 0 to 1
        
    ## EQUILIBRIA
        find_equilibrium_point: finds equilibrium points for a given list
        
    ## DICTIONARY FUNCTIONS
        merge_two_dicts: merges two dictionaries together
    
    ## DISTANCES BETWEEN ATOMS [ NOTE: These do not yet account for periodic boundary conditions! ]
        calc_xyz_dist_matrix: calculates xyz distance matrix given the coordinates
        calc_dist2_btn_pairs: calculates distance^2 between two pairs (taken from md.traj's numpy distances)
        calc_total_distance2_matrix: calculates total distance matrix^2. Note, distance^2 is useful if you only care about the minimum / maximum distances (avoiding sqrt function!)
            In addition, this function prevents numpy memory error by partitioning the atoms list based on how many you have. This is important for larger system sizes.
            
    ## SIMILARITY FUNCTIONS
        common_member_length: calculate common members between two arrays
    
CREATED ON: 02/27/2018

AUTHOR(S): 
    Alex K. Chew (alexkchew@gmail.com)
    Brad C. Dallin (enter Brad's email address here)

UPDATES:
    20180326 - BCD - added find_residue_names function
    20180327 - AKC - added find_total_atoms function
    20180328 - AKC - add split_traj_function, which splits the trajectory and calculates to prevent memory error
    20180409 - AKC - added unit_vector and angle_between functions, which can help with vector algebra
    20180413 - AKC - added finding equilibrium point function, which is great for assessing equilibria for plots
    20180420 - AKC - added calculation of xyz distance matrix and between pairs
    20180622 - AKC - added rescale_vector function that can rescale arrays from 0 to 1
                    - added "create_atom_pairs_list", which can quickly create atom pairs list
    20180625 - AKC - added functionality to "find_atom_index", where it does not need a residue name. It can look for atom index based purely on atom names
    20180627 - AKC - added functionality "find_water_index" that can find water index within a trajectory
    20180628 - AKC - added functionality "create_atom_pairs_with_self", which can quickly create atom pairs for gold-gold, etc.
    20180823 - AKC - added functionality "calc_pair_distances_between_two_atom_index_list" which calculates pair distances between atom index lists
    20181115 - AKC - added functionality "split_general_functions" which calculates data based on splitting
"""

### IMPORTING FUNCTIONS
import time
import numpy as np
import MDDescriptors.core.initialize as initialize # Checks file path
import mdtraj as md

### FUNCTION TO FIND TOTAL RESIDUES AND THE INDEX OF THOSE RESIDUES
def find_total_residues(traj, resname):
    '''
    This function takes your residue name and finds the residue indexes and the total number of residues
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

### FUNCTION TO FIND THE INDEX OF ATOM NAMES
def find_atom_index(traj, atom_name, resname = None):
    '''
    The purpose of this function is to find the atom index based on the residue name and atom name
    INPUTS:
        traj: trajectory from md.traj
        atom_name: [str] Name of your atom, e.g. 'O2'
        resname: [str, OPTIONAL, default = None] Name of your residue, e.g. 'HOH'
            NOTE: If resname is None, then this will look for all atomnames that match "atom_name" variable
    OUTPUTS:
        atom_index: [list] atom index corresponding to the residue name and atom name
    '''
    if resname is None:
        atom_index = [ each_atom.index for each_atom in traj.topology.atoms if each_atom.name == atom_name]
    else:
        atom_index = [ each_atom.index for each_atom in traj.topology.atoms if each_atom.residue.name == resname and each_atom.name == atom_name]
    return atom_index

### FUNCTION TO FIND TOTAL ATOMS AND THE INDEX OF THOSE ATOMS GIVEN RESIDUE NAME
def find_total_atoms(traj, resname):
    '''
    This function takes your residue names and find the atom indexes and the total number of atoms
    INPUTS:
        traj: [class] A trajectory loaded from md.load
        resname:[str] residue name, e.g. 'HOH'
    OUTPUTS:
        num_atoms:[int] Total number of atoms
        atom_index:[list] index of the atoms
    '''
    ## FINDING ATOM INDEX
    atom_index = [ each_atom.index for each_atom in traj.topology.atoms if each_atom.residue.name == resname]
    ## FINDING TOTAL NUMBER OF ATOMS
    num_atoms = len(atom_index)
    return num_atoms, atom_index

### FUNCTION TO FIND RESIDUE NAMES
def find_residue_names( traj, ):
    '''
    The purpose of this function is to find the residue names of the molecules 
    within a MD trajectory
    
    Inputs
    ------
    traj : md.traj
        A trajectory loaded from md.load
    
    Outputs
    -------
    res_name : list of strings of all the residue names : [List] : dtype=string
    
    '''
    return list(set([ residue.name for residue in traj.topology.residues ]))

## FUNCTION TO FIND ALL ATOM TYPES FOR A GIVEN RESIDUE
def find_specific_atom_index_from_residue( traj, residue_name, atom_type = 'O' ):
    '''
    The purpose of this function is to find all atom indexes of a type of a specific residue. 
    For instance, we would like to see find all the oxygens for a given residue.        
    INPUTS:
        traj: [md.traj]
            trajectory file from md.load
        residue_name: [str] 
            name of the residue
        atom_type: [str, default = 'O']
            atom type you are interest in. Use the chemical symbol
        
    '''
    ## FINDING ALL RESIDUE INDICES
    residue_index = [ x.index for x in traj.topology.residues if x.name == residue_name]
    ## FINDING ALL ATOM INDICES IF MATCHING
    atom_index = [ atom.index for each_residue_index in residue_index 
                             for atom in traj.topology.residue(each_residue_index).atoms 
                             if atom.element.symbol == atom_type ]
    return atom_index

### FUNCTION TO FIND TOTAL SOLUTES AND RESIDUES GIVEN A TRAJECTORY
def find_multiple_residue_index( traj, residue_name_list ):
    '''
    The purpose of this function is to find multiple residue indices and total number of residues given a list of residue name list
    INPUTS:
        traj: [md.traj]
            trajectory from md.traj
        residue_name_list: [list]
            residue names in a form of a list that is within your trajectory
    OUTPUTS:
        total_residues: [list]
            total residues of each residue name list
        residue_index: [list]
            list of residue indices
    '''
    # CREATING EMPTY ARRAY TO STORE
    total_residues, residue_index = [], []
    # LOOPING THROUGH EACH POSSIBLE SOLVENT
    for each_solvent_name in residue_name_list:
        ## FINDING TOTAL RESIDUES
        each_solvent_total_residue, each_solvent_residue_index= find_total_residues(traj, resname=each_solvent_name)
        ## STORING
        total_residues.append(each_solvent_total_residue)
        residue_index.append(each_solvent_residue_index)
    return total_residues, residue_index

### FUNCTION TO FIND ATOM NAMES
def find_atom_names(traj, residue_name):
    '''
    The purpose of this function is to find the atom names given the residue name
    INPUTS:
        traj: trajectory file from md.load
        residue_name: [STRING] name of the residue
    OUTPUTS:
        atom_names: [LIST] list of strings of the atom names
    '''
    residue_index = [ x.index for x in traj.topology.residues if x.name == residue_name][0]
    atom_names = [ atom.name for atom in traj.topology.residue(residue_index).atoms ]
    return atom_names

### FUNCTION TO FIND UNIQUE RESIDUE NAMES
def find_unique_residue_names(traj):
    ''' 
    This function simply finds all the residues in your trajectory and outputs its unique residue name
    INPUTS:
        traj: trajectory from md.traj
    OUTPUTS:
        List of unique residues
    '''
    return list(set([ residue.name for residue in traj.topology.residues ]))

### FUNCTION TO FIND RESIDUE / ATOM INDICES GIVEN RESIDUE NAME AND ATOM NAMES
def find_residue_atom_index(traj,residue_name = 'HOH', atom_names = None):
    '''
    The purpose of this function is to look at your trajectory's topology and find the atom index that you care about.
    INPUTS:
        traj: trajectory from md.traj
        residue_name: residue name as a string (i.e. 'HOH')
        atom_names: [str, default = None]
            list of atom names within your residue (i.e. ['O','H1','H2'])
            If None, then just find all possible atoms from your residue index
    OUTPUTS:
        residue_index: list of residue indices
        atom_index: list of atom indices
    '''
    ## FINDING ALL RESIDUES OF THE TYPE
    residue_index = [ x.index for x in traj.topology.residues if x.name == residue_name]
    ## FINDING ALL ATOMS THAT YOU CARE ABOUT
    if atom_names == None:
        atom_index = [ [ atom.index for atom in traj.topology.residue(res).atoms ] for res in residue_index ]
    else:
        atom_index = [ [ atom.index for atom in traj.topology.residue(res).atoms if atom.name in atom_names ] for res in residue_index ]
    
    return residue_index, atom_index


### FUNCTION TO COMPUTE MASS BASED ON RESIDUE INDEX
def calc_mass_from_residue_name(traj, residue_name = 'HOH' ):
    '''
    The purpose of this function is to compute the mass of residues from trajectory, given that you know one of of the residue names
    INPUTS:
        traj: [object]
            trajectory from md.traj
        residue_name: [str, default='HOH']
            residue name as a string (i.e. 'HOH')
    OUTPUTS:
        total_mass: [float]
            total mass in grams
    '''

    residue_index, atom_index = find_residue_atom_index(traj, residue_name)
    ## FINDING MASS OF A SINGLE GROUP
    atom_mass = [ traj.topology.atom(atom_ind).element.mass for atom_ind in atom_index[0]]
    ## FINDING TOTAL MASS
    total_mass = np.sum(atom_mass)
    return total_mass

### FUNCTION TO FIND CENTER OF MASS OF THE RESIDUES / ATOMS
def find_center_of_mass( traj, residue_name = 'HOH', atom_names = ['O','H1','H2'] ):
    '''
    The purpose of this function is to find the center of mass of your residues given the residue name and atom names. Note that atom names is a list. 
    INPUTS:
        traj: trajectory from md.traj
        residue_name: residue name as a string (i.e. 'HOH')
        atom_names: list of atom names within your residue (i.e. ['O','H1','H2'])
    OUTPUTS:
        center_of_mass: Numpy array of the center of masses ( NUM FRAMES X NUM ATOMS X 3 (X,Y,Z))
    NOTES: This function may have issues later due to the high vectorization approach!
    '''    
    ## KEEPING TRACK OF TIME
    COM_time=time.time()
    ### INITIALIZATION OF COM
    ## FINDING ATOM AND RESIDUE INDICES
    residue_index, atom_index = find_residue_atom_index(traj, residue_name, atom_names)
    ## FINDING MASS OF A SINGLE GROUP
    atom_mass = [ traj.topology.atom(atom_ind).element.mass for atom_ind in atom_index[0]]
    ## FINDING TOTAL MASS
    totalMass = np.sum(atom_mass)
    print("--- COM CALCULATION FOR %s FRAMES, %s RESIDUE (%s residues), and %s ATOM TYPES (%s atoms) ---" %(len(traj),
                                                                                   residue_name,
                                                                                   len(residue_index),
                                                                                   atom_names,
                                                                                   len(residue_index)*len(atom_names)
                                                                                   ) )
    
    ### COM CALCULATION
    # FINDING POSITION OF ALL ATOMS
    position_all_atoms = traj.xyz[:, atom_index] # Frame x atom x positions
    ## GETTING SHAPE OF ALL POSITIONS
    n_frames, n_residues, n_atoms, n_coordinates = position_all_atoms.shape
    ## REPLICATING MASSES FOR MATRIX MULTIPLICATION
    rep_mass = np.tile(np.transpose(atom_mass).reshape((n_atoms,1)), (n_frames,n_residues,1, 1)) # 1 for x,y,z coordinates already defined
    ## MULTIPLYING TO GET M_i * x_i
    multiplied_numerator = position_all_atoms * rep_mass
    ## SUMMING ALL M_i * X_i
    summed_numerator = np.sum(multiplied_numerator, axis=2 ) # Summing within each of the residues
    ## DIVIDING NUMERATOR TO GET COM
    center_of_mass = summed_numerator / totalMass
    ## PRINTING TOTAL TIME TAKEN
    h, m, s = initialize.convert2HoursMinSec( time.time()-COM_time )
    print('Total time for COM calculation was: %d hours, %d minutes, %d seconds \n' %(h, m, s))
    
    return center_of_mass

### FUNCTION TO CALCULATE THE ENSEMBLE VOLUME OF A TRAJECTORY
def calc_ensemble_vol( traj ):
    '''
    The purpose of this function is to take your trajectory and find the ensemble average volume. This is assuming your box is a cubic one.
    INPUTS:
        traj: trajectory
    OUTPUTS:
        ensemVol: Ensemble volume, typically nm^3
    '''
    # List of all unit cell lengths
    unitCellLengths = traj.unitcell_lengths
    unitCellVolumes = unitCellLengths * unitCellLengths * unitCellLengths # Assuming cubic
    # Now, using numpy to find average
    vol = np.mean(unitCellVolumes)
    return vol

### FUNCTION TO CREATE ATOM PAIRS LIST
def create_atom_pairs_list(atom_1_index_list, atom_2_index_list ):
    '''
    The purpose of this function is to create all possible atom pairs between two lists. Note that we use numpy to speed up atom pair generation list
    This function is especially useful when you are generating atom lists for distances, etc.
    Note that this function is way faster than using list comprehensions! (due to numpy using C++ to quickly do computations)
    INPUTS:
        atom_1_index_list: [np.array, shape=(N,1)] index list 1, e.g. [ 0, 1, 4, .... ]
        atom_2_index_list: [np.array, shape=(N,1)] index list 2, e.g. [ 9231, ...]
    OUTPUTS:
        atom_pairs: [np.array, shape=(N_pairs,2)] atom pairs when varying index 1 and 2
            e.g.:
                [ [0, 9231 ],
                  [1, 9321 ], ...
                  ]
    '''
    ## CREATING MESHGRID BETWEEN THE ATOM INDICES
    xv, yv = np.meshgrid( atom_1_index_list,  atom_2_index_list)

    ## STACKING THE ARRAYS
    array = np.stack( (xv, yv), axis = 1)

    ## TRANSPOSING ARRAY
    array_transpose = array.transpose(0, 2, 1) #  np.transpose(array)

    ## CONCATENATING ALL ATOM PAIRS
    atom_pairs = np.concatenate(array_transpose, axis=0)
    ## RETURNS: (N_PAIRS, 2)
    return atom_pairs

### FUNCTION TO CREATE ATOM PAIRS WITH SELF ATOMS, SUCH AS GOLD-GOLD, OR ANY OTHER STRUCTURAL OBJECTS
def create_atom_pairs_with_self(indices):
    '''
    The purpose of this script is to create atom pairs for a set of atoms with itself. For example, you may want the atom indices of gold atoms to gold atoms, but do not want the distance calculations to repeat.
    This script is useful for those interested in generating atom pairs for a list with itself.
    INPUTS:
        indices: [np.array, shape=(num_atoms, 1)] Index of the atoms
    OUTPUTS:
        atom_pairs: [np.array, shape=(N_pairs,2)] atom pairs when varying indices, but NO repeats!
        e.g.: Suppose you had atoms [0, 1, 2], then the list will be:
            [[0, 1],
             [0, 2],
             [1, 2]]
            NOTE: There are no repeats in the atom indices here! In addition, an atom cannot interact with itself!
        upper_triangular: [np.array] indices of the upper triangular matrix, which you can use to create matrix
    '''
    ## FINDING NUMBER OF ATOMS
    num_atoms = len(indices)
    
    ## DEFINING A UPPER TRIANGLAR MATRIX
    upper_triangular = np.triu_indices(num_atoms, k = 1)
    
    ## FINDING ATOM INDICES
    atom_indices = np.array(upper_triangular).T

    ## CORRECTING ATOM INDICES BASED ON INPUT ATOM INDEX
    atom_pairs = indices[atom_indices]
    
    return atom_pairs, upper_triangular


### FUNCTION TO FIND THE DISTANCES FOR SINGLE FRAME
def calc_pair_distances_with_self_single_frame(traj, atom_index, frame=-1, periodic = True):
    '''
    The purpose of this function is to quickly calculate the pair distances based on a trajectory of coordinates
    NOTES: 
        - This function finds the pair distances based on the very last frame! (editable by changing frame)
        - Therefore, this function only calculates pair distances for a single frame to improve memory and processing requirements
        - The assumption here is that the pair distances does not significantly change over time. In fact, we assume no changes with distances.
        - This was developed for gold-gold distances, but applicable to any metallic or strong bonded systems
    INPUTS:
        traj: trajectory from md.traj
        atom_index: [np.array, shape=(num_atoms, 1)] atom indices that you want to develop a pair distance matrix for
        frame: [int, default=-1] frame to calculate gold-gold distances
        periodic: [logical, default=True] True if you want PBCs to be accounted for
    OUTPUTS:
        distance_matrix: [np.array, shape=(num_atom,num_atom)] distance matrix of gold-gold, e.g.
        e.g.
            [ 0, 0.15, ....]
            [ 0, 0   , 0.23, ...]
            [ 0, ... , 0]
    '''
    ## FINDING TOTAL NUMBER OF ATOMS
    total_atoms = len(atom_index)
    ## CREATING ATOM PAIRS
    atom_pairs, upper_triangular_indices = create_atom_pairs_with_self( atom_index )
    ## CALCULATING DISTANCES
    distances = md.compute_distances( traj = traj[frame], atom_pairs = atom_pairs, periodic = periodic, opt = True )
    ## RESHAPING DISTANCES ARRAY TO MAKE A MATRIX
    # CREATING ZEROS MATRIX
    distances_matrix = np.zeros( (total_atoms, total_atoms)  )
    ## FILLING DISTANCE MATRIX
    distances_matrix[upper_triangular_indices] = distances[0]
    return distances_matrix

### FUNCTION TO CALCULATE DISTANCES FOR A SINGLE FRAME USING MD.TRAJ
def calc_pair_distances_between_two_atom_index_list(traj, atom_1_index, atom_2_index, periodic=True ):
    '''
    The purpose of this function is to calculate distances between two atom indices
    NOTES:
        - This function by default calculates pair distances of the last frame
        - This is designed to quickly get atom indices
        - This function is expandable to multiple frames
    INPUTS:
        traj: [class]
            trajectory from md.traj
        atom_1_index: [np.array, shape=(num_atoms, 1)] 
            atom_1 type indices
        atom_2_index: [np.array, shape=(num_atoms, 1)] 
            atom_2 type indices
        periodic: [logical, default=True] 
            True if you want PBCs to be accounted for
    OUTPUTS:
       distances: [np.array, shape=(num_frame, num_atom_1, num_atom_2)] 
           distance matrix with rows as atom_1 and col as atom_2. 
    '''
    ## FINDING TOTAL NUMBER OF ATOMS
    total_atom_1 = len(atom_1_index)
    total_atom_2 = len(atom_2_index)
    ## FINDING TOTAL FRAMES
    total_frames = len(traj)
    ## GENERATING ATOM PAIRS
    atom_pairs = create_atom_pairs_list(atom_2_index, atom_1_index)
    ''' RETURNS ATOM PAIRS LIKE: 
            [
                    [ATOM_1_IDX1, ATOM2_IDX1],
                    [ATOM_1_IDX1, ATOM2_IDX2],
                                ...
                    [ATOM_1_IDXN, ATOM2_IDXN],
            ]
    '''
    ## CALCULATING DISTANCES
    distances = md.compute_distances(
                                    traj = traj,
                                    atom_pairs = atom_pairs,
                                    periodic = periodic
            ) ## RETURNS TIMEFRAME X (NUM_ATOM_1 X NUM_GOLD) NUMPY ARRAY
    ## RESHAPING THE DISTANCES
    distances = distances.reshape(total_frames, total_atom_1, total_atom_2)
    ## RETURNS: TIMEFRAME X NUM_ATOM_1 X NUM_ATOM 2 (3D MATRIX)
    return distances
    

### FUNCTION TO FIND WATER RESIDUE INDEX AND ATOM INDEX
def find_water_index(traj, water_residue_name = 'HOH'):
    '''
    The purpose of this function is to find the residue index and atom index of water
    INPUTS:
        traj: [class] trajectory from md.traj
        water_residue_name: [str] residue name for water
    OUTPUTS:
        num_water_residues: [int] total number of water molecules
        water_residue_index: [list] list of water residue index
        water_oxygen_index: [np.array] atom list index of water oxygen
    '''        
    ## FINDING RESIDUE INDEX OF WATER
    num_water_residues, water_residue_index = find_total_residues(traj=traj, resname = water_residue_name)
    ## FINDING ALL OXYGENS INDEXES OF WATER
    water_oxygen_index = np.array(find_atom_index( traj, resname = water_residue_name, atom_name = 'O'))
    return num_water_residues, water_residue_index, water_oxygen_index

#######################################
### SPLITTING TRAJECTORY FUNCTIONS ####
#######################################

### FUNCTION THAT SPLITS THE TRAJECTORY, CALCULATES USING SOME FUNCTION, THEN OUTPUTS AS NUMPY ARRAY
# NOTE: This is useful if a trajectory is too long! 
def split_traj_function( traj, split_traj=50, input_function = None, optimize_memory = False, **input_variables):
    '''
    The purpose of this function is to split the trajectory up assuming that your input function is way too expensive to calculate via vectors
    INPUTS:
        traj: trajectory from md.traj
        input_function: input function. Note that the input function assumes to have a trajectory input. Furthermore, the output of the function is a numpy array, which will be the same length as the trajectory.
        input_variables: input variables for the function
        optimize_memory: [logical, default = False] If True, we will assume your output is a numpy array. Then, we will pre-allocate memory so you do not raise MemoryError.
            NOTE: If this does not work, then you have a difficult problem! You have way too much data to appropriately do your calculations. If this is the case, then:
                - pickle your output into segments
                - Do analysis for each segment (hopefully, you will not have to do this!)
    OUTPUTS:
        output_concatenated: (numpy array) Contains the output values from the input functions
    SPECIAL NOTES:
        The function you input should be a static function! (if within class) The main idea here is that we want to split the trajectory and calculate something. The output should be simply a numpy array with the same length as the trajectory!
    '''
    ## IMPORTING MODULES
    import sys
    import time
    ## PRINTING
    from MDDescriptors.core.initialize import convert2HoursMinSec
    ## CHECKING INPUT FUNCTION
    if input_function is None:
        print("Error in using split_traj_function! Please check if you correctly split the trajectory!")
        sys.exit()
    else:
        ## FINDING TRAJECTORY LENGTH
        traj_length = len(traj)
        ## PRINTING
        print("*** split_traj_function for function: %s ***"%(input_function.__name__))
        print("Splitting trajectories for each %d intervals out of a total of %d frames"%(split_traj, traj_length))
        ## STORING TIME
        start_time = time.time()
        ## CREATING SPLIT REGIONS
        split_regions = [[i,i + split_traj] for i in range(0, traj_length, split_traj)]
        ## SPLITTING TRAJECTORY BASED ON INPUTS
        traj_list = [ traj[regions[0]:regions[1]] for regions in split_regions]
        # traj_list = [traj[i:i + split_traj] for i in range(0, traj_length, split_traj)]
        ## CREATING BLANK LIST
        if optimize_memory == False:
            output_storage = []
        else:
            print("Optimization memory has been enabled! Creating empty array to fit your matrix!")
        ## LOOPING THROUGH THE TRAJECTORY
        for index, current_traj in enumerate(traj_list):
            ## KEEP TRACK OF CURRENT TIME
            current_time = time.time()
            ## PRINTING AND KEEPING TRACK OF TIME
            print("%s: WORKING ON TRAJECTORIES %d ps TO %d ps OUT OF %d ps"%(input_function.__name__,current_traj.time[0], current_traj.time[-1], traj.time[-1]))
            ## RUNNING INPUT FUNCTIONS
            output = input_function(current_traj, **input_variables)
            
            ## STORING OUTPUT TO CORRESPONDING OUTPUT STORAGE SPACE
            if optimize_memory == False:
                output_storage.append(output)
            else:
                ## IF FIRST FRAME, CREATE THE MATRIX
                if index == 0: 
                    ## FINDING SHAPES
                    output_shape = output.shape[1:] ## OMITTING TIME
                    ## FINDING FULL SHAPE
                    full_shape = tuple([traj_length] + list(output_shape))
                    print("CREATING MATRIX OF ARRAY SIZE: %s"%(', '.join([str(each) for each in full_shape ]) ) )
                    ## CREATING EMPTY ARRAY
                    output_storage = np.empty( full_shape )
                ## STORING ARRAY
                output_storage[ split_regions[index][0]:split_regions[index][1], : ] = output[:]
                                
            h, m, s = convert2HoursMinSec(time.time() - current_time)
            ## PRINTING TOTAL TIME
            print("---------> %d hours, %d minutes, %d seconds"%(h,m,s))
        if optimize_memory == False:
            ## FINALLY, CONCATENATING TO COMBINE ALL THE OUTPUT
            output_storage = np.concatenate(output_storage, axis=0)
        ## WRITING TOTAL TIME
        h, m, s =convert2HoursMinSec( time.time() - start_time)
        print('TOTAL TIME ELAPSED FOR %s: %d hours, %d minutes, %d seconds  '%(input_function.__name__, h, m, s))
        ##TODO: can re-adjust script to increase intervals if it feels confident -- could be a while loop criteria
    return output_storage


### FUNCTION THAT SPLITS A LIST INTO MULTIPLE PARTS
def split_list(alist, wanted_parts=1):
    '''
    The purpose of this function is to split a larger list into multiple parts
    INPUTS:
        alist: [list] original list
        wanted_parts: [int] number of splits you want
    OUTPUTS:
        List containing chunks of your list
    Reference: https://stackoverflow.com/questions/752308/split-list-into-smaller-lists?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
    '''
    length = len(alist)
    return [ alist[i*length // wanted_parts: (i+1)*length // wanted_parts] 
             for i in range(wanted_parts) ]

### FUNCTION THAT SPLITS THE TRAJECTORY, RUNS SOME CALCULATIONS, THEN TAKES AVERAGE AND STD OF EACH CALCULATION
# NOTE: This is great for calculating averaging and std
def split_traj_for_avg_std( traj, num_split, input_function, split_variables_dict, **static_variables):
    '''
    The purpose of this script is to split a trajectory into multiple parts, use a function to do some calculations, then average/std the results
    INPUTS:
        traj: trajectory from md.traj
        num_split: number of times to split the trajectory
        input_function: input function. Note that the input function assumes to have a trajectory input.
        split_variables_dict: variables that you WANT to split. This for example could be center of masses across trajectory (needs to be split!)
            NOTE: This is a dictionary. The dictionary should coincide with the input names of the function
            e.g.:
                input_split_vars = { 'COM_Solute'   : self.rdf.solute_COM,                          # Solute center of mass
        **static_variables: variables that does not change when you split the trajectory
            e.g.
                input_static_vars = { 'solute_res_index'    : self.rdf.solute_res_index,                        # Residue index for the solute
    OUTPUTS:
        output: results as a form of a list
    '''
    ## PRINTING
    print("----- split_traj_avg_std -----")
    print("WORKING ON TRAJECTORY WITH TIME LENGTH OF: %d ps"%(traj.time[-1] - traj.time[0]))
    print("SPLITTING TRAJECTORY IN %d PIECES:"%(num_split))
    ## SPLITTING THE TRAJECTORY
    split_traj = split_list(traj, num_split)
    ## SPLITTING THE VARIABLES
    for each_variable in split_variables_dict.keys():
        print("--> SPLITTING VARIABLE %s"%(each_variable))
        split_variables_dict[each_variable] = split_list(split_variables_dict[each_variable], num_split)
    
    ## CREATING LIST TO STORE OUTPUT
    output = []
    
    ## LOOPING THROUGH EACH TRAJECTORY
    for index, each_traj in enumerate(split_traj):
        ## PRINTING
        print("WORKING ON TRAJECTORY: %s : %d ps to %d ps"%(input_function.__name__, each_traj.time[0], each_traj.time[-1] ))
        ## GETTING THE VARIABLES
        current_split_variables_dict = {key:value[index] for (key,value) in split_variables_dict.items()}
        ## NOW, INPUTTING TO FUNCTION
        current_output = input_function( each_traj,**merge_two_dicts(current_split_variables_dict, static_variables) )
        ## STORING OUTPUT
        output.append(current_output)

    return output
    
### FUNCTION THAT SPLITS DICTIONARY BASED ON TRAJECTORY AND RUNS VARIABLES
def split_general_functions( input_function, split_variable_dict, static_variable_dict, num_split = 1  ):
    '''
    The purpose of this function is to split a trajectory, corresponding variables, and run the function again. The outputs of the functions will be stored into a list.
    INPUTS:
        input_function: [function]
            input function. Note that the input function assumes you will import split variables and static variables
        num_split: [int, default = 1]
            number of times to split the trajectory. Default =  1 means no splitting
        split_variables_dict: variables that you WANT to split. This for example could be center of masses across trajectory (needs to be split!)
            NOTE: This is a dictionary. The dictionary should coincide with the input names of the function
            e.g.:
                input_split_vars = { 'COM_Solute'   : self.rdf.solute_COM,                          # Solute center of mass
        static_variables_dict: variables that does not change when you split the trajectory
            e.g.
                input_static_vars = { 'solute_res_index'    : self.rdf.solute_res_index,                        # Residue index for the solute
    OUTPUTS:
        output: [list]
            output in a form of a list. Note that if you had multiple arguments, it will output as a list of tuples.
    '''
    ## SPLITTING VARIABLES
    for each_variable in split_variable_dict.keys():
        print("--> SPLITTING VARIABLE %s"%(each_variable))
        split_variable_dict[each_variable] = split_list(split_variable_dict[each_variable], num_split)
        
    ## CREATING LIST TO STORE THE OUTPUTS
    output = []
    
    ## LOOPING THROUGH THE SPLITTED FILES
    for index in range(num_split):
        ## GETTING VARIABLES AND COMBINING
        current_split_variables_dict = {key:value[index] for (key,value) in split_variable_dict.items()}
        ## MERGING TWO DICTIONARIES
        merged_dicts = merge_two_dicts(current_split_variables_dict, static_variable_dict)
        ## RUNNING INPUT FUNCTION AND STORING
        output.append( input_function( **merged_dicts ) )
        
    return output

### FUNCTION TO CALCULAGE AVG AND STD ACCORDINGLY
def calc_avg_std(list_of_dicts):
    '''
    The purpose of this script is to calculate the average and standard deviation of several values.
    ASSUMPTION:
        We are assuming that the input is a list of dictionaries:
            [{'var1': 2}, {'var1':3} ] <-- We want to average var1, etc.
        Furthermore, we are assuming that each dictionaries should have more or less the same keys (otherwise, averaging makes no sense!)
    INPUTS:
        list_of_dicts: [list] List of dictionaries
    OUTPUTS:
        avg_std_dict: [dict] Dictionary with the same keys, but each key has the following:
            'avg': average value
            'std': standard deviation
    '''
    avg_std_dict = {}
    ## LOOPING THROUGH THE DICTIONARY
    for dict_index, each_dict in enumerate(list_of_dicts):
        ## GETTING THE KEYS
        current_keys = each_dict.keys()
        ## LOOPING THROUGH EACH KEY 
        for each_key in current_keys:
            ## ADDING KEY IF IT IS NOT INSIDE
            if each_key not in avg_std_dict.keys():
                avg_std_dict[each_key] = [list_of_dicts[dict_index][each_key]]
            ## APPENDING IF WE ALREADY HAVE THE KEY
            else:
                avg_std_dict[each_key].append(list_of_dicts[dict_index][each_key])
    
    ## ADD THE END, TAKE AVERAGE AND STANDARD DEVIATION
    avg_std_dict = {key: {'avg': np.mean(value), 'std':np.std(value)} for (key,value) in avg_std_dict.items()}
    return avg_std_dict

### FUNCTION TO CALCULATE AVG AND STANDARD DEVIATION OF A VARIABLE ACROSS MULTIPLE FRAMES
def calc_avg_std_of_list( traj_list ):
    '''
    The purpose of this script is to calculate the average and standard deviation of a list (e.g. a value that fluctuates across the trajectory)
    INPUTS:
        traj_list: [np.array or list] list that you want average and std for
    OUTPUTS:
        avg_std_dict: [dict] dictionary with the average ('avg') and standard deviation ('std')
    NOTES:
        - This function takes into account 'nan', where non existent numbers are not considered in the mean or std
    '''
    ## TESTING IF LIST HAS NON EXISTING NUMBERS (NANS). IF SO, AVERAGE + STD WITHOUT THE NANs
    if np.any(np.isnan(traj_list)) == True:
        ## NON EXISTING NUMBERS EXIST
        avg = np.nanmean(traj_list)
        std = np.nanstd(traj_list)
    else:
        ## FINDING AVERAGE AND STANDARD DEVIATION
        avg = np.mean(traj_list)
        std = np.std(traj_list)
    
    ## STORING AVERAGE AND STD TO A DICTIONARY
    avg_std_dict = { 'avg': avg,
                     'std': std,}
    return avg_std_dict
    
####################################
########## VECTOR ALGEBRA ##########
####################################

### FUNCTION TO CONVERT A VECTOR OF ANY LENGTH TO A UNIT VECTOR
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

### FUNCTION TO GET THE ANGLE BETWEEN TWO VECTORS IN RADIANS
def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
        Uses dot product , where theta = arccos ( unitVec(A) dot unitVec(B))
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)) 

### FUNCTION TO RESCALE VECTOR BETWEEN 0 AND 1
def rescale_vector(vec):
    '''
    The purpose of this function is to rescale a vector between 0 and 1
    INPUTS:
        vec: [np.array, shape=(N,1)] numpy array that you want to be normalized
    OUTPUTS:
        rescaled_vec: [np.array, shape=(N,1)] numpy array that has been rescaled between 0 and 1
    '''
    ## FINDING NEW VECTOR USING MINIMA AND MAXIMA
    rescaled_vec =  ( vec - np.min(vec) ) / (np.max(vec) - np.min(vec))
    return rescaled_vec

### FUNCTION TO FIND EQUILIBRIUM POINT
## EXAMPLE: Suppose you have a radial distribution function and you want to find when it equilibrated. We do this by:
#   - Reverse the list of the y-values
#   - Find when the average is off a tolerance
def find_equilibrium_point(ylist=[],tolerance=0.015 ):
    '''
    The purpose of this function is to take your y-values, and find some equilibrium point based on some tolerance. This does a running average and sees if the value deviates too far.
    INPUTS:
        ylist: yvalues as a list
        tolerance: tolerance for your running average
    OUTPUTS:
        index_of_equil: Index of your ylist where it has equilibrated
    '''
    # Reversing list
    ylist_rev = list(reversed(ylist[:]))
    
    # Pre-for loop
    # Starting counter
    counter = 1
    endPoint = len(ylist_rev)
    
    # Assuming initial values
    runningAvg = ylist_rev[0] # First value
    nextValue = ylist_rev[counter] # Next Value
    
    while abs(runningAvg-nextValue) < tolerance and counter < endPoint - 1:
        # Finding new running average
        runningAvg = (runningAvg * counter + nextValue) / (counter + 1)
        
        # Adding to counter
        counter += 1
               
        # Finding the next energy
        nextValue = ylist_rev[counter]
        
    # Going back one counter, clearly the one that worked last
    correct_counter = counter - 1
    
    # Getting index of the correct list
    index_of_equil = endPoint - correct_counter - 1 # Subtracting 1 because we count from zero
    
    return index_of_equil    


### FUNCTION TO MERGE TWO DICTIONARIES
## THIS IS REQUIRED FOR PYTHON 2 TO 3.4
def merge_two_dicts(x, y):
    '''
    The purpose of this function is to merge two dictionaries
    INPUTS:
        x, y: dictionaries
    OUTPUTS:
        z: merged dictionary
    '''
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z

### FUNCTION TO CALCULATE XYZ DISTANCE MATRIX BETWEEN ALL COORDINATE ATOMS
def calc_xyz_dist_matrix(coordinates):
    '''
    The purpose of this script is to take coordinates and find the difference between i and j
    INPUTS:
        Coordinates - Numpy array
    OUTPUTS:
        deltaXarray - Array for x-differences (j-i)
        deltaYarray - Array for y-differences (j-i)
        deltaZarray - Array for z-differences (j-i)
    '''    
    def makeDistArray(Vector):
        '''
        This script simply takes a vector of x's, y's, or z's and creates a distance matrix for them
        INPUTS:
            Vector - A list of x coordinates for example
        OUTPUTS:
            Array - Distance matrix j - i type
        '''
        vectorSize = len(Vector)
        
        # Start by creating a blank matrix
        myArray = np.zeros( (vectorSize, vectorSize ) )
        
        # Use for-loop to input into array and find distances
        for i in range(0,vectorSize-1):
            for j in range(i,vectorSize):
                myArray[i,j] = Vector[j]-Vector[i]
        
        return myArray # - myArray.T

    deltaXarray = makeDistArray(coordinates.T[0])  # X-values
    deltaYarray = makeDistArray(coordinates.T[1])  # Y-values
    deltaZarray = makeDistArray(coordinates.T[2])  # Z-values
    return deltaXarray, deltaYarray, deltaZarray


### FUNCTION TO CALCULATE DISTANCE BETWEEN PAIRS (TAKEN FROM MD.TRAJ)
def calc_dist2_btn_pairs(coordinates, pairs):
    """        
    Distance squared between pairs of points in each coordinate
    INPUTS:
        coordinates: N x 3 numpy array
        pairs: M x 2 numpy array, which are pairs of atoms
    OUTPUTS:
        distances: distances in the form of a M x 1 array
    """
    delta = np.diff(coordinates[pairs], axis=1)[:, 0]
    return (delta ** 2.).sum(-1) # ** 0.5 

### FUNCTION TO CALCULATE TOTAL DISTANCE MATRIX
def calc_total_distance2_matrix(coordinates, force_vectorization=False, atom_threshold=2000):
    '''
    This function calls for calc_xyz_dist_matrix and simply uses its outputs to calculate a total distance matrix that is squared
    INPUTS:
        coordinates: numpy array (n x 3)
        force_vectorization: If True, it will force vectorization every time
        atom_threshold: threshold of atoms, if larger than this, we will use for loops for vectorization
    OUTPUTS:
        dist2: distance matrix (N x N)
    '''
    ## FINDING TOTAL LENGTH OF COORDINATES
    total_atoms = len(coordinates)
    
    ## SEEING IF WE NEED TO USE LOOPS FOR TOTAL DISTANCE MATRIX
    if total_atoms < atom_threshold:
        num_split=0
    else:
        num_split = int(np.ceil(total_atoms / atom_threshold))
        
    if num_split == 0 or force_vectorization is True:
        deltaXarray, deltaYarray, deltaZarray = calc_xyz_dist_matrix(coordinates)
        # Finding total distance^2
        dist2 = deltaXarray*deltaXarray + deltaYarray*deltaYarray + deltaZarray*deltaZarray
    else:
        print("Since number of atoms > %s, we are shortening the atom list to prevent memory error!"%(atom_threshold))
        ## SPLITTING ATOM LIST BASED ON THE SPLITTING
        atom_list = np.array_split( np.arange(total_atoms), num_split )
        ## CREATING EMPTY ARRAY
        dist2 = np.zeros((total_atoms, total_atoms))
        total_atoms_done = 0
        ## LOOPING THROUGH EACH ATOM LIST
        for index, current_atom_list in enumerate(atom_list):
            ## FINDING MAX AN MINS OF ATOM LIST
            atom_range =[ current_atom_list[0], current_atom_list[-1] ]
            ## FINDING CURRENT TOTAL ATOMS
            current_total_atoms = len(current_atom_list)
            ## PRINTING
            print("--> WORKING ON %d ATOMS, ATOMS LEFT: %d"%(current_total_atoms, total_atoms - total_atoms_done) )
            ## GETTING ATOM PAIRS
            pairs = np.array([[y, x] for y in range(atom_range[0], atom_range[1]+1 ) for x in range(y+1, total_atoms) ] )
            ## CALCULATING DISTANCES
            current_distances = calc_dist2_btn_pairs(coordinates, pairs)
            ## ADDING TO TOTAL ATOMS
            total_atoms_done += current_total_atoms
            ## STORING DISTANCES
            dist2[pairs[:,0],pairs[:,1]] = current_distances
            
            ##TODO: inclusion of total time
            ##TODO: optimization of atom distances
    return dist2

##########################################
########## SIMILARITY FUNCTIONS ##########
##########################################
### FUNCTION TO FIND THE LENGTH OF TOTAL MEMBERS
def common_member_length(a, b):
    return len(np.intersect1d( a, b ))    

### FUNCTION TO FLATTEN LIST OF LIST
def flatten_list_of_list( my_list ):
    ''' This flattens list of list '''
    return [item for sublist in my_list for item in sublist]

