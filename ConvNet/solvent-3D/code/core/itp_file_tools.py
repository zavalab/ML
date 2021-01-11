# -*- coding: utf-8 -*-
"""
itp_file_tools.py
This script stores all itp file functions

Created on: 07/31/2019

FUNCTIONS:
    find_itp_file_given_res_name: locates all itp file given a residue name
    find_oxygen_bonding_info: locates all bonding information to oxygens
    find_hydroxyl_groups_from_oxygen_bonding: locates all hydroxyl group information based on oxygen bonding information

Written by:
    - Alex K. Chew (alexkchew@gmail.com)
"""

## IMPORTING MODULES
import sys
import core.read_write_tools as read_write_tools
import glob
import numpy as np
    
## IMPORTING FUNCTIONS
from core.import_tools import read_file_as_line

### FUNCTION TO FIND ITP FILE GIVEN RESIDUE NAME
def find_itp_file_given_res_name(directory_path,
                                 residue_name_list,
                                 verbose = True):
    '''
    This function finds the itp file given residue name list
    INPUTS:
        directory_path: [str]
            directory path to itp files
        residue_name_list: [list]
            list of residue name
    OUTPUTS:
        itp_file_list: [list]
            list of itp file with itp information
    '''
    ## LOCATING ITP FILES
    itp_files = glob.glob( directory_path + '/*.itp' )
    
    ## DEFINIGN LIST
    itp_file_list = []
    
    ## LOOPING TO FULL PATH TO ITP FILES
    for full_itp_path in itp_files:
        if verbose is True:
            print("CHECKING ITP FILE: %s"%(full_itp_path) )
        try:
            itp_info = read_write_tools.extract_itp(full_itp_path)
            ## STORE ONLY IF THE RESIDUE NAME MATCHES THE LIGAND
            if itp_info.residue_name in residue_name_list:
                ## PRINTING
                if verbose is True:
                    print("Storing itp file for residue name: %s"%( itp_info.residue_name ) )
                ## APPENDING
                itp_file_list.append(itp_info)
                break ## Breaking out of for loop
        except Exception: # <-- if error in reading the itp file (happens sometimes!)
            pass
        
    ## SEEING IF LENGTH IS ZERO
    if len(itp_file_list) == 0:
        print("Error! No itp file found!")
        print("Check the find_itp_file_given_res_name function.")
        print("Stopping here! You may receive errors for missing itp file!")
        sys.exit()
        
    return itp_file_list

### FUNCTION TO FIND OXYGEN BONDING DETAILS
def find_oxygen_bonding_info(itp_file,
                             verbose = True):
    '''
    The purpose of this function is to find the atoms bonded to oxygen given 
    that you have the itp information.
    INPUTS:
        itp_file: [obj]
            itp file objects
        verbose: [logical]
            True if you want to print verbosely
    OUTPUTS:
        oxygen_bonding_array: [np.array]
            numpy array of oxygen bonding information
    '''
    if verbose is True:
        print("--- Locating all oxygen bonding information for %s ---"%(itp_file.residue_name))
    ## FINDING ALL OXYGEN ATOM NAMES
    oxygen_names = sorted(list(set([ eachAtom for eachBond in itp_file.bonds_atomname for eachAtom in eachBond if 'O' in eachAtom])))
    
    ## BONDING ARRAY
    bonds_atomname_array = np.array(itp_file.bonds_atomname)
    
    ## STORING
    oxygen_bonding_array = []
    
    ## LOOPING THROUGH OXYGEN NAMES
    for each_oxygen in oxygen_names:
        ## DEFINING ALL BONDED ATOMNAMES
        bonded_atomnames = bonds_atomname_array[np.any(bonds_atomname_array == each_oxygen, axis=1),:]
        ## FINDING ATOMNAMES NOT EQUAL TO OXYGENS
        bonded_atomnames_without_oxygen = bonded_atomnames[bonded_atomnames != each_oxygen]
        ## STORING
        oxygen_bonding_array.extend(bonded_atomnames)
        ## PRINTING
        if verbose is True:
            print("Atoms bonded to %s: %s"%(each_oxygen, bonded_atomnames_without_oxygen ))
        
    ## CONVERTING TO ARRAY
    oxygen_bonding_array = np.array(oxygen_bonding_array)
    
    return oxygen_bonding_array

### FUNCTION TO FIND HYDROXYL GROUPS FROM BONDING INFORMATION
def find_hydroxyl_groups_from_oxygen_bonding(oxygen_bonding_array,
                                             verbose = True):
    '''
    This function finds the hydroxyl group from oxygen bonding information
    INPUTS:
        oxygen_bonding_array: [np.array, shape=(N,2)]
            oxygen bonding array, e.g.
                array([['C5', 'O1'],
                       ['O1', 'H8'],
                       ['C4', 'O2'],
                       ['O2', 'H9'],...
        verbose: [logical]
            True if you want to print verbosely
    OUTPUTS:
        hydroxyl_bonding_array: [np.array]
            hydroxyl bonding details
    '''
    if verbose is True:
        print("--- Locating all hydroxyl information given oxygen array ---")
    ## STORING
    hydroxyl_bonding_array = []
    
    ## LOCATING HYDROGEN BONDING INFORMATION
    for each_bond in oxygen_bonding_array:
        ## SEE IF THE BOND HAS ONE OXYGEN AND ONE HYDROGEN
        if 'H' in each_bond[0] and 'O' in each_bond[1] or \
           'H' in each_bond[1] and 'O' in each_bond[0]:
               ## APPENDING
               hydroxyl_bonding_array.append(each_bond)
               ## PRINTING
               if verbose is True:
                   print("Hydroxyl combination found for: %s"%(each_bond) )
    
    ## STORING AS ARRAY
    hydroxyl_bonding_array = np.array(hydroxyl_bonding_array)
    
    return hydroxyl_bonding_array

### FUNCTION TO JUST READ THE ITP RESIDUE NAME
def read_residue_name_from_itp(itp_path):
    '''
    The purpose of this function is to simply read the residue name from an 
    itp file. 
    INPUTS:
        itp_path: [str]
            path to itp file
    OUTPUTS:
        res_name: [str]
            name of the residue
    '''
    ## READING FILE
    file_info = read_file_as_line(file_path = itp_path, verbose = False)
    ## CLEANING ITP FILE
    itp_data = [ eachLine.replace('\t', ' ') for eachLine in file_info if not eachLine.startswith(";") ]
    ## FINDING MOLECULE TYPE INFORMATION
    molecule_type_line = [ idx for idx,each_line in enumerate(itp_data) if '[ moleculetype ]' in each_line ][0]
    ## EXTRACTING RESIDUE NAME
    res_name =  itp_data[molecule_type_line + 1].split()[0]
    
    return res_name
#%%
## MAIN FUNCTION
if __name__ == "__main__":
    ## DEBUGGING
    itp_path=r"R:/scratch/SideProjectHuber/Simulations/190925-4ns_mixed_solvent_with_FRU_HMF/mdRun_403.15_6_nm_FRU_10_WtPercWater_spce_ethylenecarbonate/ethylenecarbonate.itp"
    res_name = read_residue_name_from_itp(itp_path)
