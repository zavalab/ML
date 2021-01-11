# -*- coding: utf-8 -*-
"""
generate_grid_interpolation.py
The purpose of this code is to generate a grid interpolation of a solvent system.
The idea here is we want to represent our solvent system in a way that can be 
represented for a machine learning approach. 

Created on: 03/14/2019

FUNCTIONS:
    normalize_3d_rdf: normalizes 3D RDFs

Written by:
    - Alex K. Chew (alexkchew@gmail.com)

"""


### IMPORTING MODULES
import numpy as np
import sys
import os
import mdtraj as md
import pickle # Used to store variables    
from datetime import datetime

## CUSTOM FUNCTIONS
import core.import_tools as import_tools # Loading trajectory details
import core.calc_tools as calc_tools # Loading calculation tools
import core.itp_file_tools as itp_file_tools ## ALL ITP FILE TOOL INFORMATION
from core.plotting_scripts import plot_voxel
from core.check_tools import check_path_to_server

### FUNCTION TO NORMALIZE VOXEL
def normalize_3d_rdf(num_dist, bin_volume, total, volume, ):
    '''
    The purpose of this function is to normalize a 3D RDF. Here, we assume that 
    you have computed all the number distribution information, e.g. x, y, z, and N, where
    N is the number of solvents within that voxel. 
    INPUTS:
        num_dist: [np.array, shape=(time, x, y, z)]
            Number distribution as a function of time
        bin_volume: [float]
            bin volume
        total: [float]
            total possible in your box, e.g. total solvent atoms
        volume: [np.array, shape=(time,1)]
            Volume of frame
    OUTPUTS:
        normalized_num_dist: [np.array, shape=(time, x, y, z)]
            normalized number distribution. Normalization is based on the following:
                (num / bin volume) / ( total num possible / volume)
        NOTE: Swap axis required to correctly divide num distribution with the volume
    '''
    ## FINDING NORMALIZATION DENSITY
    overall_num_density =  total / volume
    ## NORMALIZING NUMBER DISTRIBUTION BY BIN VOLUME
    normalized_num_dist = num_dist / bin_volume
    ## THEN NORMALIZING BY TOTAL DENSITY
    normalized_num_dist = np.swapaxes((np.swapaxes(normalized_num_dist, 0, 3) ) / overall_num_density, 0, 3 )
    return normalized_num_dist

### CLASS FUNCTION TO GENERATE GRID INTERPOLATION ARRAY PER FRAME
class generate_grid_interpolation:
    '''
    The purpose of this function is to generate grid interpolation array. The algorithm 
    is as follows:
        - Generate an array to store all the variables
        - Use histogram methods to find all the atoms
        - Map the data into an array
    INPUTS:
        traj_data: [obj]
            trajectory data
        solute_name: [str]
            solute name as a string
        map_box_size: [float]
            map box size in nms
        map_box_increment: [float]
            increment box size
        map_type: [str, default='allatom']
            all atom mapping type
        verbose: [logical, default =False ]
            True if you want the function to print information
        normalization: [str, default='maxima']
            'maxima' if you want RGB channels to be normalized from 0 to 1 by dividing maxima
        debug: [logical, default=False]
            True if you want to save information
    OUTPUTS:
        ## INPUt VARIABLES
    
    '''
    ### INITIALIZING
    def __init__(self, 
                 traj_data, 
                 solute_name = 'tBuOH',
                 solvent_name = ['HOH'],
                 map_box_size = 4,
                 map_box_increment = 0.1,
                 map_type = 'allatom',
                 normalization = 'maxima',
                 verbose = False,
                 debug = False,
                 ):
        ## STORING INITIAL INFORMATION
        self.solute_name = solute_name
        self.solvent_name = solvent_name
        self.map_box_size = map_box_size
        self.map_box_increment = map_box_increment
        self.map_type = map_type
        self.normalization = normalization
        self.verbose = verbose
        
        ### DEFINING TRAJECTORY
        traj = traj_data.traj
        
        ### PRINTING
        if self.verbose is True:
            print("**** CLASS: generate_grid_interpolation ****")
            
        ### CHECK IF SOLUTE EXISTS IN TRAJECTORY
        if self.solute_name not in traj_data.residues.keys():
            print("ERROR! Solute (%s) not available in trajectory. Stopping here to prevent further errors. Check your 'Solute' input!")
            sys.exit()
        
        ### CHECK SOLVENT NAMES TO SEE IF THEY EXISTS IN THE TRAJECTORY
        self.solvent_name = [ each_solvent for each_solvent in self.solvent_name if each_solvent in traj_data.residues.keys() ]
        
        ### AVERAGE VOLUME
        self.average_volume = np.mean(traj_data.traj.unitcell_volumes)
        
        ### FINDING TOTAL FRAMES
        self.total_frames = len(traj)
        
        ### FINDING TOTAL RESIDUES
        self.num_solvent_residues = [traj_data.residues[each_solvent] for each_solvent in self.solvent_name]
        
        ### FINDING TOTAL ATOMS
        self.num_solvent_atoms = [ calc_tools.find_total_atoms(traj,each_solvent)[0] for each_solvent in self.solvent_name ]
        
        ### FINDING BOX LIMITS
        self.find_box_range()
        
        ## DEFINING RESIDUE NAME LIST
        self.residue_name_list = [self.solute_name] + self.solvent_name
        
        ## FINDING HYDROXYL BONDING
        if self.map_type == '3channel_hydroxyl' or self.map_type == '4chan_hydroxyl':
            self.find_hydroxyl_atomnames(path_to_itp = traj_data.directory )
        
        ### FINDING CENTER OF MASSES OF SOLUTE
        print("\n---- CALCULATING CENTER OF MASS ----\n")
        self.COM_solute = self.find_solute_COM( traj=traj, residue_name=self.solute_name)
        ### FINDING SOLUTE COM AVG AND STD
        # self.COM_solute_avg = np.mean(self.COM_solute, axis = 0) # Averaging center of mass
        # self.COM_solute_std = np.std(self.COM_solute, axis = 0) # Standard deviation of the COM
        
        ## COMPUTING SOLUTE COM DISTANCE
        self.calc_solute_displacements(traj = traj)
        
        ## CHECK IF CENTER OF MASSES NEED TO BE CALCULATED
        self.COM_solvent = [[] for i in range(len(self.solvent_name))] # Empty list
        if self.map_type == 'COM':
            ### FINDING CENTER OF MASS OF SOLVENT
            for index, solvent in enumerate(self.solvent_name):
                self.COM_solvent[index] = self.find_solvent_COM( traj= traj, residue_name= solvent )
            
        ### FINDING DISPLACEMENTS FOR EACH SOLVENT
        self.displacements = [] # Empty list
        for index, solvent in enumerate(self.solvent_name):
            self.displacements.append( self.calc_solute_solvent_displacements( traj= traj,
                                                                              solute_name = self.solute_name,
                                                                              # solute_center_of_mass = self.COM_solute_avg,
                                                                              solute_center_of_mass = self.COM_solute,
                                                                              solvent_name = self.solvent_name[index],
                                                                              solvent_center_of_mass = self.COM_solvent[index],
                                                                              ) )
        
        ### USING HISTOGRAM METHODS TO CALCULATE PROBABILITY DISTRIBUTION
        self.num_dist = {} # Empty dictionary
        
        ## COMPUTING DISPLACEMENTS FOR SOLUTE
        if self.map_type != '3channel_oxy' and \
           self.map_type != '3channel_hydroxyl' and \
           self.map_type != 'solvent_only' and \
           self.map_type != '4chan_hydroxyl':
            self.num_dist[self.solute_name] = self.calc_num_dist( 
                                                       solvent_name = self.solute_name,
                                                       displacements = self.solute_displacements,
                                                       )
        
        ## COMPUTING DISPLACEMENTS FOR THE SOLVENT
        for index, solvent in enumerate(self.solvent_name):
            self.num_dist[solvent] = self.calc_num_dist( 
                                                       solvent_name = self.solvent_name[index],
                                                       displacements = self.displacements[index]
                                                       )
            
        ## FINDING DISPLACEMENTS OF OXYGEN ONLY
        if self.map_type == 'allatomwithsoluteoxygen' or \
           self.map_type == '3channel_oxy' or \
           self.map_type == '3channel_hydroxyl' or \
           self.map_type == '4chan_hydroxyl':
            ## NAME FOR SOLUTE WITH OXYGEN
            if self.map_type != '3channel_hydroxyl' and self.map_type != '4chan_hydroxyl':
                self.solute_oxygen_name = self.solute_name + '_oxygens'
            else:
                self.solute_oxygen_name = self.solute_name + '_hydroxyl'
            ## COMPUTING NUMBER DISTRIBUTION
            self.num_dist[self.solute_oxygen_name] = self.calc_num_dist( 
                                                                   solvent_name = self.solute_oxygen_name,
                                                                   displacements = self.solute_oxy_displacements,
                                                                   )
            
        ## COMPUTING DISPLACEMENTS OF ALL NON OXYGENS
        if self.map_type == '4chan_hydroxyl':
            ## DEFINING NAME
            self.solute_non_atoms = self.solute_name + '_nonhydroxyl'
            ## COMPUTING NUMBER DISTRIBUTION
            self.num_dist[self.solute_non_atoms] = self.calc_num_dist( 
                                                                   solvent_name = self.solute_non_atoms,
                                                                   displacements = self.solute_nonoxy_displacements,
                                                                   )
            
                    
            
        ## NORMALIZING
        self.normalize_grid(traj_data = traj_data)
        
        ## FINDING COSOLVENT NAME
        # print(self.solvent_name)
        try:
            cosolvent_name = [each_residue for each_residue in self.solvent_name if each_residue != 'HOH'][0]
        except IndexError:
            print("Error! No cosolvent found!")
            print("Here are all the solvent names:")
            print("%s"%(', '.join(self.solvent_name)) )
            print("Perhaps you have missed the solvents in the POSSIBLE_SOLVENTS list")
            print("Check: core > global_vars, POSSIBLE_SOLVENTS variables")
        
        
        ## CONGLOMERTING GRID DATA RESULTS
        # self.grid_rgb_data= np.zeros( (self.total_frames, *self.bin_array) + (3,))
        
        if self.map_type == 'allatomwithsoluteoxygen' or self.map_type == '4chan_hydroxyl': # oxygen as the fourth channel
            self.grid_rgb_data= np.zeros( tuple( [self.total_frames] + list(self.bin_array) ) + (4,))
        elif self.map_type == 'solvent_only': # oxygen as the fourth channel
            ## ONLY SOLVENTS
            self.grid_rgb_data= np.zeros( tuple( [self.total_frames] + list(self.bin_array) ) + (2,))
        else: # default
            self.grid_rgb_data= np.zeros( tuple( [self.total_frames] + list(self.bin_array) ) + (3,))
        
        ## DEFINING RGB COLORS
        ## FIRST CHANNEL
        self.grid_rgb_data[..., 0] = self.num_dist['HOH'][:] # Water is red
        
        ## SECOND CHANNEL
        if self.map_type == '3channel_oxy' or self.map_type == '3channel_hydroxyl' or self.map_type == '4chan_hydroxyl':
            ## ADD OXYGENS AS 2ND CHANNEL
            self.grid_rgb_data[..., 1] = self.num_dist[self.solute_oxygen_name][:] # Solute is green
        else:
            if self.map_type != 'solvent_only':
                ## ADDING SOLUTE AS SECOND CHANNEL
                self.grid_rgb_data[..., 1] = self.num_dist[self.solute_name][:] # Solute is green
            else:
                ## ADDING COSOLVENT AS THE SECOND CHANNEL
                self.grid_rgb_data[..., 1] = self.num_dist[cosolvent_name][:] # Cosolvent is blue
        
        ## THIRD CHANNEL -- COSOLVENT IF WE INCLUDE REACTANT
        if self.map_type != 'solvent_only':
            self.grid_rgb_data[..., 2] = self.num_dist[cosolvent_name][:] # Cosolvent is blue
        
        ## FOURTH CHANNEL
        if self.map_type == 'allatomwithsoluteoxygen': # oxygen as the fourth channel
            self.grid_rgb_data[..., 3] = self.num_dist[self.solute_oxygen_name][:] # Cosolvent is blue
        elif self.map_type == '4chan_hydroxyl':
            self.grid_rgb_data[..., 3] = self.num_dist[self.solute_non_atoms][:] # Cosolvent is blue
        
        ## CLEARING DATA 
        if debug is False:
            self.clear_data()
        
        return
    
    ### FUNCTION TO NORMALIZE THE DATA
    def normalize_grid(self, traj_data):
        '''
        The purpose of this function is to normalize the number distribution data.
            self: class object
            traj: [md.traj]
                trajectory object
        '''
        if self.normalization == 'rdf':
            ## NEED TO FIND ALL VOLUMES
            volumes = traj_data.traj.unitcell_volumes
            ## NEED TO FIND ALL SOLVENT RESIDUES
            total_atoms={}
            ### FINDING TOTAL ATOMS
            total_atoms[self.solute_name] = calc_tools.find_total_atoms(traj_data.traj,self.solute_name)[0]
            for index, solvent in enumerate(self.solvent_name):
                total_atoms[solvent] = calc_tools.find_total_atoms(traj_data.traj,solvent)[0]
            ### NORMALIZING THE GRID
            self.num_dist[self.solute_name] = normalize_3d_rdf( num_dist = self.num_dist[self.solute_name],
                                                                bin_volume = self.bin_volume,
                                                                total = total_atoms[self.solute_name],
                                                                volume = volumes,
                                                               )
            ## NORMALIZE SOLVENTS
            for index, solvent in enumerate(self.solvent_name):
                ### NORMALIZING THE GRID
                self.num_dist[solvent] = normalize_3d_rdf( num_dist = self.num_dist[solvent],
                                                                    bin_volume = self.bin_volume,
                                                                    total = total_atoms[solvent],
                                                                    volume = volumes,
                                                                   )
        return
        
    
    ### FUNCTION TO CLEAR DATA
    def clear_data(self):
        ''' This function clears data if you are saving into pickle to save memory '''
        self.displacements = []
        self.num_dist = []
        self.COM_solute_avg = []
        self.COM_solute_std = []
        self.solute_positions = []
        self.solute_displacements = []
        
    ### FUNCTION TO CALCULATE HISTOGRAM INFORMATION
    def calc_num_dist( self, solvent_name, displacements, freq = 1000 ):
        '''
        The purpose of this function is to take the displacements and find a histogram worth of data
        INPUTS:
            self: class object
            solvent_name: Name of the solvent
            displacements: numpy vector containing all displacements
            freq: Frequency of frames you want to print information
        OUTPUTS:
            grid_storage: [np.array, shape=(total_frames, num_bins, num_bins,num_bins)]
                grid storage that is normalized by the maximum of the grid
        '''
        ## PRINTING
        print("\n--- GENERATING HISOGRAM DATA FOR SOLVENT: %s ---"%(solvent_name))
        ## FINDING RANGE BASED ON CENTER OF MASS
        arange = (self.r_range, self.r_range, self.r_range)
        
        ### TESTING NUMPY HISTOGRAM DD
        grid, edges = np.histogramdd(np.zeros((1, 3)), bins=(self.max_bin_num, self.max_bin_num, self.max_bin_num), range=arange, normed=False)
        grid *=0.0 # Start at zero
        
        ### GENERATING EMPTY NUMPY ARRAY
        grid_storage = np.zeros( ( self.total_frames, self.max_bin_num, self.max_bin_num, self.max_bin_num) ) 
        
        ## CHECKING IF DISPLACEMENTS IS NOT EMPTY
        if displacements.size != 0:
            for frame in range(self.total_frames):
                ### DEFINING DISPLACEMENTS
                current_displacements = displacements[frame]
                ### USING HISTOGRAM FUNCTION TO GET DISPLACEMENTS WITHIN BINS
                hist, edges = np.histogramdd(current_displacements, bins=(self.bin_array), range=arange, normed=False)
                ### ADDING TO GRID
                if self.normalization == 'maxima':
                    grid_storage[frame] = hist[:] / np.max(hist)
                else:
                    grid_storage[frame] = np.copy(hist[:])
                
                ### PRINTING
                if frame % freq == 0:
                    ### CURRENT FRAME
                    print("Generating histogram for frame %s out of %s, found total atoms: %s"%(frame, self.total_frames, np.sum(hist)))
                    ''' BELOW IS A WAY TO CHECK
                    ### CHECKING
                    x_displacements = current_displacements[:,0]; y_displacements = current_displacements[:,1]; z_displacements = current_displacements[:,2]
                    ### FINDING NUMBER OF WATER MOLECULES WITHIN A SINGLE FRAME
                    count = (x_displacements > arange[0][0]) & (x_displacements < arange[0][1]) & \
                    (y_displacements > arange[1][0]) & (y_displacements < arange[1][1]) & \
                    (z_displacements > arange[2][0]) & (z_displacements < arange[2][1])
                    print("Checking total count: %s"%(np.sum(count)))
                '''
        else:
            print("Since displacement size is zero for %s, setting grid to zeros"%(solvent_name))

        return grid_storage

    ### FUNCTION TO FIND THE BOX RANGE
    def find_box_range(self):
        '''
        The purpose of this function is to take the input data and find the bounds of the box
        INPUTS:
            self: class object
        OUTPUTS:
            self.max_bin_num: maximum bin number (integer)
            self.map_half_box_length: half box length in nm
            self.plot_axis_range: plotting range if you plotted in Cartesian coordinates, tuple (min, max)
            self.r_range: Range that we are interested in as a tuple (-half_box, +half_box)
            self.bin_volume: [float]
                bin volume
        '''
        ## FINDING MAXIMUM NUMBER OF BINS
        self.max_bin_num = int(np.floor(self.map_box_size / self.map_box_increment))
        ## FINDING HALF BOX LENGTH
        self.map_half_box_length = self.map_box_size/2.0
        ## FINDING PLOT AXIS RANGE
        self.plot_axis_range = (0, self.max_bin_num) # Plot length for axis in number of bins
        ## FINDING RANGE OF INTEREST
        self.r_range = (-self.map_half_box_length, self.map_half_box_length)
        ## DEFINING BIN ARRAY
        self.bin_array = (self.max_bin_num, self.max_bin_num, self.max_bin_num)
        ## FINDING BIN VOLUME
        self.bin_volume = self.map_box_increment**3
        
        return
        
    ## ASSUMES SINGLE SOLUTE
    def find_solute_COM( self, traj, residue_name ):
        '''
        This function simply finds the center of mass given the residue name
        INPUTS:
            self: class object
            traj: traj from md.traj
            residue_name: Name of residue as a string
        OUTPUTS:
            center_of_mass: center of mass as a numpy arrays
        '''
        ## FINDING ALL RESIDUES OF THE TYPE
        residue_index = [ x.index for x in traj.topology.residues if x.name == residue_name]
        atom_index = [ x.index for x in traj.topology.atoms if x.residue.index in residue_index ]
        ## FINDING ATOM NAMES
        atom_names = [ traj.topology.atom(current_atom_index).name for current_atom_index in atom_index]
        ## FINDING SOLUTE COM
        center_of_mass = calc_tools.find_center_of_mass(traj=traj, residue_name=residue_name, atom_names=atom_names)
        return center_of_mass
        
    ### FUNCTION TO FIND SOLUTE DISPLACEMENTS
    def calc_solute_displacements(self, traj):
        '''
        The purpose of this function is to compute the solute displacement from the 
        center of mass.
        INPUTS:
            self:
                class object
            traj: [obj]
                trajectory from md.traj
        OUTPUTS:
            self.solute_displacements: [np.array]
                solute displacements
        '''
        ## FINDING ALL ATOM INDICES OF THE SOLUTE
        residue_index, atom_index = calc_tools.find_residue_atom_index( traj = traj, 
                                                                        residue_name = self.solute_name)
        ## FINDING ALL POSITIONS
        self.solute_positions = traj.xyz[:, atom_index] # Frame x atom x positions
        
        ## REDEFINING SOLUTE POSITIONS (ASSUMINE ONE SOLUTE)
        self.solute_positions = self.solute_positions[:,0,:,:]
        
        ## FINDING DISPLACEMENTS FROM CENTER OF MASS
        self.solute_displacements =  self.solute_positions - self.COM_solute # Updated 06/17/2019 -- correcting for COM measurements
        # self.solute_displacements =  self.solute_positions - self.COM_solute_avg
        # grid_interpolation.solute_positions - grid_interpolation.COM_solute
        # - self.COM_solute <-- updated with average (05/30/2019)
        # -- does not make a lot of sense to do this without solvents referencing to different points as well
    
        ## ASSUMING ONE SOLUTE
        atom_index = atom_index[0]
        
        ## FINDING DISPLACEMENTS OF OXYGEN ONLY
        if self.map_type == 'allatomwithsoluteoxygen' or \
           self.map_type == '3channel_oxy' or \
           self.map_type == '3channel_hydroxyl' or \
           self.map_type == '4chan_hydroxyl':
            ## START BY FINDING ALL INDICES
            if self.map_type != '3channel_hydroxyl' and self.map_type != '4chan_hydroxyl':
                self.solute_oxygen_index = calc_tools.find_specific_atom_index_from_residue(traj = traj, 
                                                                                       residue_name = self.solute_name, 
                                                                                       atom_type = 'O',   )
            else:
                ## GETTING ALL ATOM NAMES
                atom_names = np.unique(self.hydroxyl_bonding_array.flatten()).tolist()
                ## GETTING ATOM NAMES FROM TRAJ
                atom_names_traj = calc_tools.find_atom_names(traj, residue_name=self.solute_name)
                ## MATCHING ATOM INDEXES TO ATOM NAMES
                self.solute_oxygen_index = []
                ## PRINTING
                print("--- Finding each atom within hydroxyls index ---")
                for each_index, each_atom_name in enumerate(self.itp_file.atom_atomname):
                    ## SEEING IF WITHIN LIST
                    if each_atom_name in atom_names:
                        print("Storing atom name %s -- traj name: %s"%(each_atom_name, atom_names_traj[each_index] ) )
                        self.solute_oxygen_index.append(atom_index[each_index])                
                
                ## PRINTING
                print("------------------------------------------------")
                
#                ## HYDROXYL GROUPS
#                solute_oxygen_index = calc_tools.find_residue_atom_index(traj = traj,
#                                                                         residue_name = self.solute_name,
#                                                                         atom_names = atom_names)[1][0] ## ASSUMING ONE RESIDUE
                ## CHECKING IF LENGTH MAKES SENSE
                if len(atom_names) != len(self.solute_oxygen_index):
                    print("Error! Atom names of hydroxyl groups do not match atom index!")
                    print("Total atom names: %d"%( len(atom_names) ))
                    print("Total atom index: %d"%( len(self.solute_oxygen_index) ))
                    print("This could lead to errors of missing atoms from atom index")
            
            ## FINDING INDEX THAT MATCHES
            indexes_matched = [ atom_index.index(each_index) for each_index in self.solute_oxygen_index]
            if self.map_type == '4chan_hydroxyl':
                ## FINDING A RANGE OF ATOM INDEX
                atom_index_range = np.arange(len(atom_index))
                ## FINDING INDEXES THAT DO NOT MATCH
                index_not_matched = [ each_value for each_value in atom_index_range if each_value not in indexes_matched ]
                # print(indexes_matched)
                # print(index_not_matched)
                # np.nonzero(np.isin( atom_index_range, indexes_matched, invert=True ))[0]
                ## STORING
                self.solute_nonoxy_displacements = self.solute_displacements[:,index_not_matched,:]
            
            ## DEFINING DISPLACEMENTS
            self.solute_oxy_displacements = self.solute_displacements[:,indexes_matched,:]
        
        return
    
    ### FUNCTION TO FIND DISPLACEMENTS BETWEEN SOLUTE AND SOLVENT CENTER OF MASSES
    def calc_solute_solvent_displacements(self, 
                                          traj, 
                                          solute_name, 
                                          solute_center_of_mass, 
                                          solvent_name, 
                                          solvent_center_of_mass,
                                          periodic = True):
        '''
        This function calculates the displacements between solute and solvent center of masses using md.traj's function of displacement. First, the first atoms of the solute and solvent are found. Then, we copy the trajectory, change the coordinates to match center of masses of solute and solvent, then calculate displacements between solute and solvent.
        INPUTS:
            traj: trajectory from md.traj
            solute_name: name of the solute as a string
            solute_center_of_mass: numpy array containing the solute center of mass
            solvent_name: name of the solvent as a string
            solvent_center_of_mass: numpy array containing solvent center of mass
        OUTPUTS:
            displacements: displacements as a time frame x number of displacemeny numpy float           
        '''
        print("--- CALCULATING SOLUTE(%s) and SOLVENT(%s) DISPLACEMENTS"%(solute_name, solvent_name))
        ## COPYING TRAJECTORY
        copied_traj=traj[:]
        
        ## FINDING FIRST ATOM INDEX FOR ALL SOLUTE AND SOLVENT
        Solute_first_atoms = self.find_first_atom_index(traj = traj, residue_name = solute_name )
        Solvent_first_atoms = self.find_first_atom_index(traj = traj, residue_name = solvent_name )
        
        ## CHANGING POSITION OF THE SOLUTE
        copied_traj.xyz[:, Solute_first_atoms] = solute_center_of_mass[:]
        
        if self.map_type == 'COM':
            ## CHANGING POSITIONS OF SOLVENT 
            copied_traj.xyz[:, Solvent_first_atoms] = solvent_center_of_mass[:]        
            ## CREATING ATOM PAIRS BETWEEN SOLUTE AND SOLVENT COM
            atom_pairs = [ [Solute_first_atoms[0], x] for x in Solvent_first_atoms]
        else: ## ALL OTHER OPTIONS
            #self.map_type == 'allatom' or self.map_type == 'allatomwithsoluteoxygen' or self.map_type == '3channel_oxy':
            ## FINDING ALL ATOMS
            num_solvent, solvent_index = calc_tools.find_total_atoms(traj, solvent_name)
            ## CREATING ATOM PAIRS BETWEEN SOLUTE AND EACH SOLVENT
            atom_pairs = [ [Solute_first_atoms[0], x] for x in solvent_index]
            
        ## FINDING THE DISPLACEMENTS USING MD TRAJ -- Periodic is true
        displacements = md.compute_displacements( traj=copied_traj, atom_pairs = atom_pairs, periodic = periodic)
        
        return displacements
    
    ### FUNCTION TO FIND THE FIRST ATOM OF EACH RESIDUE
    @staticmethod
    def find_first_atom_index(traj, residue_name):
        '''
        The purpose of this script is to find the first atom of all residues given your residue name
        INPUTS:
            traj: trajectory from md.traj
            residue_name: name of your residue
        OUTPUTS:
            first_atom_index: first atom of each residue as a list
        '''
        ## FINDING ALL RESIDUES
        residue_index = [ x.index for x in traj.topology.residues if x.name == residue_name]
        ## FINDING ATOM INDEX OF THE FIRST RESIDUE
        atom_index = [ [ atom.index for atom in traj.topology.residue(res).atoms] for res in residue_index ]
        ## FINDING FIRST ATOM FOR EACH RESIDUE
        first_atom_index = [ atom[0] for atom in atom_index ]
        
        return first_atom_index
    ### FUNCITON TO PLOT
    def plot_grid_rgb(self, frame =0, want_renormalize = False):
        ''' This plots grid rgb '''
        fig, ax = plot_voxel(grid_rgb_data = self.grid_rgb_data, frame=frame, want_renormalize = want_renormalize)
        return fig, ax
    
    ### FUNCTION TO STORE ALL INFORMATION WITHIN A PICKLE FILE
    def store_pickle(self, pickle_file_path=os.path.join(os.getcwd(), "pickle.pickle")):
        '''
        The purpose of this function is to store all the results into a pickle file
        INPUTS:
            self: class property
        OUTPUTS:
            pickle file within the pickle directory under the name of the class used
        '''
        ## PRINTING TOTAL SIZE OF THE OBJECT
        # print("Total size of saving pickle is: %d bytes or %d MB"%(sys.getsizeof(self), sys.getsizeof(self) / 1000000.0 ))
        ## DEPRECIATED, see size by: https://stackoverflow.com/questions/449560/how-do-i-determine-the-size-of-an-object-in-python
        with open(pickle_file_path, 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump([self.grid_rgb_data], f, protocol=2)  # <-- protocol 2 required for python2   # -1
        print("Data collection was complete at: %s\n"%(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        return

    ### FUNCTION TO PLOT GRID RGB DATA
    def plot_rgb_data_per_index( self, frame = 0, index = 1):
        '''
        The purpose of this function is to plot rgb data per index
        INPUTS:
            frame: [int]
                frame that you are interested in
            index: [int]
                index that you want, e.g. 0 is water, 1 is cosolvent, etc.
        OUTPUTS:
            fig, ax: figure and axis of rgb data plotted per frame
        '''
        ## IMPORTING MODULE
        import matplotlib.pyplot as plt
        ## DEFINING RGB DATA
        grid_rgb_data = grid_interpolation.grid_rgb_data

        ## DEFINING INDICES, E.G. 1 TO 20
        # r, g, b = np.indices(np.array(grid_rgb_data[frame][...,0].shape)+1)
        ## DEFINING GRID SHAPE
        grid_shape=np.array(grid_rgb_data[frame][...,0].shape)
        
        ## ADDING ONE TO GRID SHAPE
        x, y, z = np.indices( grid_shape + 1 ) # 
        
        ## DEFINING DATA TO PLOT
        grid_rgb_data_to_plot = grid_rgb_data[0][...,index]
        
        ## DEFINING VOXELS > 0
        voxels = grid_rgb_data_to_plot > 0
        
        # combine the color components
        colors = np.zeros( tuple(grid_shape) + (3,) ) # 20 x 20 x 20 x 3 array
        
        ## CHANGING COLOR INDEX
        if index == 0:
            color_index = 0 # Red
        elif index == 1:
            color_index = 1 # Green
        elif index == 2:
            color_index = 2 # Blue
        elif index == 3:
            color_index = 1 # Green
        else:
            color_index = 0 # Red by default
        
        colors[..., color_index] = grid_rgb_data_to_plot[:]
        # colors[..., 1] = 0
        # colors[..., 2] = 0
    
        ## PLOTTING
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        
        ## PLOTTING VOXEL
        ax.voxels(x, y, z ,voxels,
                  # facecolors = grid_rgb_data_to_plot,
                  facecolors=colors,
                  edgecolors=np.clip(2*colors - 0.5, 0, 1),  # brighter
                  linewidth=0.5) # 0.5
        
        ## SETTING AXIS LABEL
        ax.set(xlabel='x', ylabel='y', zlabel='z')
        
        return fig, ax
    
    ### FUNCTION TO FIND HYDROXYL BONDING ARRAY
    def find_hydroxyl_atomnames(self, path_to_itp):
        '''
        The purpose of this function is to find all hydroxyl atomnames 
        for a solute residue.
        INPUTS:
            path_to_itp: [path]
                path to itp file
        OUTPUTS:
            self.oxygen_bonding_array: [np.array]
                oxygen bonding array
            self.hydroxyl_bonding_array: [np.array]
                hydroxyl bonding array
        '''
        ## DEFINING RESIDUE NAME LIST
        residue_name_list = [ self.solute_name ] 
        
        ## FINDING ITP FILE
        self.itp_file = itp_file_tools.find_itp_file_given_res_name(directory_path = path_to_itp,
                                                 residue_name_list = residue_name_list,
                                                 )[0] # Only one itp file will match
        
        ## GETTING OXYGEN BONDING ARRAY
        self.oxygen_bonding_array = itp_file_tools.find_oxygen_bonding_info(self.itp_file,
                                                                            verbose = True)
        
        ## GETTING HYDROXYL BONDING ARRAY
        self.hydroxyl_bonding_array = itp_file_tools.find_hydroxyl_groups_from_oxygen_bonding(oxygen_bonding_array = self.oxygen_bonding_array,
                                                                                              verbose = True)
        
        return

#%%
if __name__ == "__main__":    
    ## DEFINING PATH INFORMATION
    analysis_dir=r"170814-7Molecules_200ns_Full" # Analysis directory
    # analysis_dir=r"190612-FRU_HMF_Run" # Analysis directory
    
    ## DEFINING SPECIFIC DIRECTORY
    specific_dir="tBuOH\\mdRun_363.15_6_nm_tBuOH_10_WtPercWater_spce_dioxane" # Directory within analysis_dir r"mdRun_363.15_6_nm_tBuOH_100_WtPercWater_spce_Pure"
    specific_dir="XYL\\mdRun_403.15_6_nm_XYL_10_WtPercWater_spce_dioxane" # Directory within analysis_dir r"mdRun_363.15_6_nm_tBuOH_100_WtPercWater_spce_Pure"
    # specific_dir="ETBE\\mdRun_343.15_6_nm_ETBE_10_WtPercWater_spce_tetrahydrofuran" # Directory within analysis_dir r"mdRun_363.15_6_nm_tBuOH_100_WtPercWater_spce_Pure"
    # specific_dir="GLU\\mdRun_393.15_6_nm_GLU_12_WtPercWater_spce_acetone" # Directory within analysis_dir r"mdRun_363.15_6_nm_tBuOH_100_WtPercWater_spce_Pure"
    
    ### DEFINING FILE NAMES
    gro_file=r"mixed_solv_prod.gro" # Structural file <-- must be a pdb file!
    xtc_file=r"mixed_solv_prod_first_20_ns_centered_with_10ns.xtc" # r"mixed_solv_prod_first_32_ns_centered.xtc"
        
    # "3channel_oxy"
    ### DEFINING PATH
    path2AnalysisDir=check_path_to_server(r"R:\scratch\SideProjectHuber\Analysis\\" + analysis_dir + '\\' + specific_dir) # PC Side
    
    ### LOADING TRAJECTORY
    traj_data = import_tools.import_traj( directory = path2AnalysisDir, # Directory to analysis
                 structure_file = gro_file, # structure file
                  xtc_file = xtc_file, # trajectories
                  )
    
    #%%
    
    ### DEFINING INPUT VARIABLES
    input_vars = {
            'traj_data'         : traj_data,
            'solute_name':             'XYL', # Solute of interest tBuOH XYL
            'solvent_name':   ['HOH', 'DIO', 'THF', 'ACE'] , # Solvent of interest HOH 'HOH' , 'GVLL'
            'map_box_size'      : 4.0, # nm box length in all three dimensions
            'map_box_increment' : 0.2, # box cell increments
            'map_type'          : '4chan_hydroxyl', 
            'normalization'     : 'maxima', # maxima, rdf
            'debug'             : True,
            }
    
    ''' Mapping types:
        - COM: center of mass of solvent is used
        - allatom: all atom (solutes) used
        - allatomwithsoluteoxygen: all atom with solute oxygens as fourth channel
        - 3channel_oxy: 3 channel, replacing reactant with oxygens of reactant
        - 3channel_hydroxyl: 3 channel, replacing reactants with hydroxyls of reactant
        - solvent_only: 2 channel, only solvents are included
        - 4chan_hydroxyl: 4 channel, hydroxyl group in one channel, the remaining atoms in another
    '''
    ## GENERATING GRID INTERPOLATION
    grid_interpolation = generate_grid_interpolation( **input_vars )
    
    #%%
    
    # test_array = [7, 8, 9, 10, 11, 17, 18, 19, 20, 21]
    
    # atom_index_array = np.arange(21)
    
    ## FINDING INDEX NOT MATCHING
    # index_not_match = np.nonzero(np.isin( grid_interpolation.atom_index_range, grid_interpolation.indexes_matched, invert=True ))
    

    
    