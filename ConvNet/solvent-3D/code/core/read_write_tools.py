# -*- coding: utf-8 -*-
"""
read_write_tools.py
This script holds all functions for reading and writing information

### CLASS DEFINITIONS
    extract_gro: extracts gro file information
    extract_itp: inputs an itp file path and extracts all details about the itp file
    extract_top: inputs topology file and extracts all information about topoloy


### GLOBAL VARIABLES:
    ITP_COMMENT_DICT: dictionary for commenting for itp files
    
### DEFINITIONS
    ## DEFINITIONS TO WRITE ITP FILES
        write_itp_type_comment: write itp type comments
        add_before_and_after_2_string: adds a string before and after
    ## ITP FILE DETAILS
        load_itp_file: loads itp file
        convert_dihedrals_to_atom_index: converts dihedral angles to atom indices
        convert_atom_index_to_elements: converts atom index to elements

Written by: Alex K. Chew (alexkchew@gmail.com, created on 02/21/2018)


*** UPDATES ***
180222 - [AKC] Added extract_gro class
180223 - [AKC] Added extract_top class
180417 - [AKC] Added read xvg class
180419 - [AKC] Edited extract itp class to correctly account for bond function
180510 - [AKC] Updated extraction of itp script
180513 - [AKC] Debugging extract_itp file --- dealing with issue of file ending and causing errors

"""
### IMPORTING MODULES
import numpy as np
import glob
import sys

### DICTIONARY FOR COMMENTS ON EACH DEFINITION
ITP_COMMENT_DICT = {
        '[ atomtypes ]'      : '; name  at.num    mass    charge   ptype          sigma(nm)      epsilon (kJ/mol) ',
        '[ nonbond_params ]' : '; i     j  func     sigma       epsilon  ',
        '[ moleculetype ]'   : '; Name   nrexcl',
        '[ atoms ]'          : ';   nr    type   resnr  residue    atom    cgnr   charge    mass',
        '[ bonds ]'          : ';  ai    aj   funct  b0   kb   ; bond properties inferred from atom types',
        '[ angles ]'         : ';  ai    aj    ak  funct  ; angle properties inferred from atom types',
        '[ dihedrals ]'      : ';  ai    aj    ak   al  funct  ; dihedral properties inferred from atom types ',
        '[ pairs ]'          : ';  ai    aj ; pairs inter 1,4 interactions',
        }

### FUNCTION TO WRITE ITP COMMENTS
def write_itp_type_comment(outputfile, itp_type):
    '''
    This function takes your itp file and the interaction type to insert a comment. This function is based on "ITP_COMMENT_DICT", which is a dictionary containing all possible comments.
    INPUTS:
        outputfile: file you want to output in (this function will add at current line)
        itp_type: what is the type you have (e.g. '[ atomtypes ]')
    OUTPUTS:
        Will output comments within outputfile
    '''
    try:
        comment = ITP_COMMENT_DICT[itp_type]
        outputfile.write("%s\n"%(comment))
    except Exception: # Do nothing -- no comments
        pass
    return

### FUNCTION TO ADD SPACES TO BEGINNING AND END OF STRING
def add_before_and_after_2_string( string, additional_string=' ' ):
    '''
    This function simply adds \n to beginning and end of your string
    INPUTS:
        string (e.g. [ bonds ])
    OUTPUTS:
        new_string (e.g. '\n[ bonds ]\n')
    '''
    new_string = additional_string + string + additional_string
    return new_string


### CLASS DEFINITION: READS TOPOLOGY FILE
class extract_top:
    '''
    This function extracts the topology file details for GROMACS.
    Written by: Alex K. Chew (alexkchew@gmail.com)
    ASSUMPTIONS:
        topology file has a '[ system ]'
        topology file has a '[ molecules ]'
    INPUTS:
        topology_path: full path to topology
    OUTPUTS:
        self.topology_path: path to topology
        self.topology_line: Lines within topology file
        ### FORCE FIELD INFORMATION
            self.line_forcefield: Line number of the force field
            self.force_field_name: Name of the force field
        ### ITP FILE INFO
            self.itp_list_index: Index for the itp files
            self.itp_list: List of itp files full line
    ### FUNCTIONS:
        find_force_field_itp: finds force field itp within topology
        find_itp_files: finds itp files within topology
        remove_comments: removes all comments from incoming data (STATIC METHOD)
        find_brackets_data: finds all lines starting with '[' (STATIC METHOD)
        
    '''
    ### INITIALIZATION
    def __init__(self, topology_path):
        print("\n--- CLASS: extract_top---")
        print("*** EXTRACTING TOP FILE: %s ***"%(topology_path))
        
        ### ADDING TOPOLOGY PATH
        self.topology_path = topology_path
        
        ### READING FILE
        with open(self.topology_path, 'r+' ) as topology_file: # For reading and writing
            ### ADDING FORCEFIELD INFO AFTER FORCEFIELD.ITP
            self.topology_lines = topology_file.readlines()
            
        ### FINDING FORCE FIELD INFORMATION
        self.find_force_field_itp()
        ### FINDING ITP FILE INFORMATION
        self.find_itp_files()
        ### CLEANING DATA
        self.topology_lines_clean = self.remove_comments(self.topology_lines)
        ### EXTRACTING DATA
        self.bracket_extract = self.find_brackets_data(self.topology_lines_clean + ['\n']) # Adding empty line at end in case you do not have one
        
        ### PRINTING SUMMARY
        print("FORCE FIELD TYPE: %s"%(self.force_field_name))
        print("NUMBER OF ITP FILES DETECTED: %s"%(len(self.itp_list)))
    
    ### FUNCTION TO FIND FORCE FIELD ITP
    def find_force_field_itp(self, forcefield_string = 'forcefield.itp' ):
        '''
        This function finds the force field information and line -- locates "forcefield.itp"
        INPUTS:
            self: class property
            forcefield_string: string that has the force field name
        OUTPUTS:
            self.line_forcefield: Line number of the force field
            self.force_field_name: Name of the force field
        '''
        self.line_forcefield = [ index for index, line in enumerate(self.topology_lines) if forcefield_string in line][0]
        ## LINE WITH FORCEFIELD.ITP
        forcefield_line = self.topology_lines[self.line_forcefield]
        ## FINDING FORCE FIELD NAME
        loc_forcefield_itp = forcefield_line.find(forcefield_string)
        try:
            loc_first_quote = forcefield_line.find('\"')
            self.force_field_name = forcefield_line[loc_first_quote+1:loc_forcefield_itp-1] # makes assumption that we have a double quote and and forward slash
        except:
            print("**Minor error** Could not correctly find %s, simplifying force field name to the entire line")
            self.force_field_name = forcefield_line
    
    ### FUNCTION TO FIND ALL ITP FILES
    def find_itp_files( self ):
        '''
        This function finds all itp files and their corresponding line numbers
        INPUTS:
            self: class property
        OUTPUTS:
            self.itp_list_index: Index for the itp files
            self.itp_list: List of itp files full line
        '''
        self.itp_list_index = [ index for index, line in enumerate(self.topology_lines) if ".itp" in line]
        self.itp_list = [ self.topology_lines[index] for index in self.itp_list_index]
        
    ### STATIC FUNCTION TO CLEAN YOUR DATA
    @staticmethod
    def remove_comments( data, comment_indicator=";"):
        '''
        This function simply takes your list of strings and remove all lines that start with ";"
        INPUTS:
            data: list of strings
        OUTPUTS:
            data_nocomments: data without comments
        '''
        data_nocomments =[ eachLine for eachLine in data if not eachLine.startswith(comment_indicator) ]
        return data_nocomments
        
    ### STATIC FUNCTION TO FIND ALL PARTS WITH BRACKETS []
    @staticmethod
    def find_brackets_data( data ):
        '''
        This function looks through the given list of data, finds all brackets, then converts them into dictionary information.
        This function uses "extractDataType", which extracts each data type
        INPUTS:
            data: the data (ideally without comments)
        OUTPUTS:
            brackets_dict: dictionary with the data
        '''
        ### FUNCTION TO EXTRACT DATA FROM ITP FILE
        def extractDataType( clean_itp, desired_type ):
            '''
            The purpose of this function is to take your itp file and the desired type (i.e. bonds) and get the information from it. It assumes your itp file has been cleaned of comments
            INPUTS:
                clean_itp: itp data as a list without comments (semicolons)
                desired_type: Types that you want (i.e. [bonds])
            OUTPUTS:
                DataOfInterest: Data for that type as a list of list
            '''
            # Finding bond index
            IndexOfExtract = clean_itp.index(desired_type)
            
            # Defining the data we are interested in
            DataOfInterest = []
            currentIndexCheck = IndexOfExtract+1 
            
            # Using while loop to go through the list to see when this thing ends
            while ('[' in clean_itp[currentIndexCheck]) is False and currentIndexCheck != len(clean_itp) - 1: # Checks if the file ended
                # Appending to the data set
                DataOfInterest.append(clean_itp[currentIndexCheck])
                # Incrementing the index
                currentIndexCheck+=1     
                
            # Removing any blanks and then splitting to columns
            DataOfInterest = [ currentLine.split() for currentLine in DataOfInterest if len(currentLine) != 0 ]
            
            return DataOfInterest
        
        # Finding all types that you can extract
        allTypes = [ eachLine for eachLine in data if '[' in eachLine ]
        
        # Finding all data
        data_by_type = [ extractDataType( clean_itp=data, desired_type=eachType) for eachType in allTypes ]
        
        # Creating dictionary for each one
        brackets_dict = {}
        for currentIndex in range(len(allTypes)):
            brackets_dict[allTypes[currentIndex]] = data_by_type[currentIndex]
        
        return brackets_dict



### CLASS DEFINITION: READS GRO FILE
class extract_gro:
    '''
    In this class, it will take the gro file and extract all the details
    Written by: Alex K. Chew (alexkchew@gmail.com)
    INPUTS:
        gro_file_path: full path to gro file
    OUTPUTS:
        ### FROM GRO FILE
        self.ResidueNum: Integer list containing residue numbers
        self.ResidueName: String name containing residue name (SOL, for example)
        self.AtomName: String containing atom name (C1, H2, etc.)
        self.AtomName_nospaces: Same as before iwthout spaces
        self.AtomNum: Integer list containing atom numbers (1, 2, 3, ...)
        self.xCoord: Float list of x-coordinates
        self.yCoord: Float list of y-coordinates
        self.zCoord: Float list of z-coordinates
        self.Box_Dimensions: List containing box dimensions
        
        ### CALCULATED/MODIFIED RESULTS
        self.unique_resid: Unique residue IDs
        self.total_atoms: total atoms
        self.total_residues: total residues
    '''

    ### INITIALIZATION
    def __init__(self, gro_file_path):
        print("\n--- CLASS: extract_gro ---")
        print("*** EXTRACTING GRO FILE: %s ***"%(gro_file_path))
        with open(gro_file_path, "r") as outputfile: # Opens gro file and make it writable
            fileData = outputfile.readlines()
    
        # Deletion of first two rows, simply the title and atom number
        del fileData[0]
        del fileData[0]

        # We know the box length is the last line. Let's extract that first.
        currentBoxDimensions = fileData[-1][:-1].split() # Split by spaces
        self.Box_Dimensions = [ float(x) for x in currentBoxDimensions ] # As floats
    
        # Deleting last line for simplicity
        del fileData[-1]
        
        # Since we know the format of the GRO file to be:
        '''
            5 Character: Residue Number (Integer)
            5 Character: Residue Name (String)
            5 Character: Atom name (String)
            5 Character: Atom number (Integer)
            8 Character: X-coordinate (Float, 3 decimal places)
            8 Character: Y-coordinate (Float, 3 decimal places)
            8 Character: Z-coordinate (Float, 3 decimal places)
        '''
        # We can extract the data for each line according to that format
        self.ResidueNum = []
        self.ResidueName = []
        self.AtomName = []
        self.AtomNum = []
        self.xCoord = []
        self.yCoord = []
        self.zCoord = []
        
        # Using for-loop to input into that specific format
        for currentLine in fileData:
            self.ResidueNum.append( int(currentLine[0:5]) )
            self.ResidueName.append( str(currentLine[5:10]) )
            self.AtomName.append( str(currentLine[10:15]) )
            self.AtomNum.append( int(currentLine[15:20]) )
            self.xCoord.append( float(currentLine[20:28]) )
            self.yCoord.append( float(currentLine[28:36]) )
            self.zCoord.append( float(currentLine[36:44]) )
        
        # CALCULATING TOTAL ATOMS
        self.total_atoms = len(self.AtomNum)
        
        # FINDING UNIQUE RESIDUE ID's
        self.unique_resid = list(set(self.ResidueNum))
        # CALCULATING TOTAL RESIDUES
        self.total_residues = len(self.unique_resid)
        
        
        ## REMOVING SPACES
        self.AtomName = [ atom.replace(" ", "") for atom in self.AtomName]
        self.ResidueName = [ res.replace(" ", "") for res in self.ResidueName]
        
        ### PRINTING SUMMARY
        print("TOTAL ATOMS: %d"%(self.total_atoms) )
        print("TOTAL RESIDUES: %d"%(self.total_residues))
        print("BOX DIMENSIONS: %s"%( [format(x, ".3f") for x in self.Box_Dimensions] ))
        
    
    
### CLASS DEFINITION: READS ITP FILE STRUCTURAL DETAILS
class extract_itp:
    '''
    In this class, it will take the itp file, and extract all the details.
    INPUTS:
        itp_file_path: full path to itp file
    OUTPUTS:
        ### GENERAL INFORMATION
        self.itp_file_path: path to itp file
        self.clean_itp: clean itp file (no comments)
        self.itp_dict: dictionary containing your different itp types
        self.residue_name: residue name
        
        ### MANDATORY INFORMATION (REQUIRED AT THE TOP OF ITP FILE)
        self.matching_mandatory: unique matching mandatory itp files
        self.total_mandatory: total mandatory itp items
        self.mandatory_items: List of list [[ MANDATORY_TYPE_STRING, INFORMATION (as a list) ]]
        
        ### MOLECULAR INFORMATION -- EMPTY IF NOT AVAILABLE
        ATOMS:
            self.atom_num: Atom numbers as a list
            self.atom_type: Atom types as a list
            self.atom_resnr: Res number as a list
            self.atom_atomname: Atomnames as a list
            self.atom_charge: Atom charges as a list
            self.atom_mass: Atom mass as a list
        BONDS:
            self.bonds_atomname: List of bonds (atom to atom names)
            self.bonds: list of bonds
            self.bonds_func: list of bonds functions
        PAIRS:
            self.pairs: List of pairs
            self.pairs_func: List of pair functions
        ANGLES:
            self.angles: List of angles (Angles x 3)
            self.angle_func: List of angle functions (Angles x 1)
        DIHEDRALS:
            self.dihedrals: List of dihedrals (dihedrals x 4)
            self.dihedrals_func: List of dihedral functions (dihedrals x 1)
        
    ACTIVE FUNCTIONS:
        find_atoms_bonded: finds atoms bound given an index
        print_itp_file: prints the itp file as is
    '''
    ### FUNCTION TO READ THE ITP FILE
    def readITP( self ):
        '''
        The purpose of this function is to read the itp file, remove the comments, and output each line as a list
        INPUTS:
            self.itp_file_path: Full path to your itp file
        OUTPUTS:
            self.clean_itp: itp data as a list without comments (semicolons)
        '''
        with open(self.itp_file_path,'r') as ITPFile:
             itp_data=ITPFile.read().splitlines()
        
        # Replacing all tabs (cleaning up)
        itp_data = [ eachLine.replace('\t', ' ') for eachLine in itp_data ]
        
        # Cleaning up itp file of all comments
        self.clean_itp =[ eachLine for eachLine in itp_data if not eachLine.startswith(";") ]
    
    ### FUNCTION TO EXTRACT ITP TO A DICTIONARY FORM
    def itp2dictionary(self):
        '''
        The purpose of this script is to take your clean itp file, then extract by types. This will also check for duplicates
        INPUTS:
            self.clean_itp: itp data as a list without comments (semicolons)
        OUTPUTS
            self.itp_dict: dictionary containing your different itp types
                NOTE, this will include duplicates under 'duplicates' key word
        FUNCTIONS:
            extractDataType: extracts all the data given a desired type
            extract_data_given_index: extracts all the data given an index
        '''
        ### FUNCTION TO EXTRACT DATA FROM ITP FILE
        def extractDataType( clean_itp, desired_type ):
            '''
            The purpose of this function is to take your itp file and the desired type (i.e. bonds) and get the information from it. It assumes your itp file has been cleaned of comments
            INPUTS:
                clean_itp: itp data as a list without comments (semicolons)
                desired_type: Types that you want (i.e. [bonds])
            OUTPUTS:
                DataOfInterest: Data for that type as a list of list
            '''
            # Finding bond index
            IndexOfExtract = clean_itp.index(desired_type)
            
            # Defining the data we are interested in
            DataOfInterest = []
            currentIndexCheck = IndexOfExtract+1 
            # Using while loop to go through the list to see when this thing ends
            while (('[' in clean_itp[currentIndexCheck]) == False and '' != clean_itp[currentIndexCheck]): # -1 Checks if the file ended  and currentIndexCheck <= len(clean_itp)
                # Appending to the data set
                DataOfInterest.append(clean_itp[currentIndexCheck])
                # Incrementing the index
                currentIndexCheck+=1
                ## IF REACHING THE END OF THE FILE, THEN LET'S MAKE SURE THE LAST DATA POINT IS INCLUDED
                if currentIndexCheck >= len(clean_itp)-1:
                    ## TRYING TO ADD LAST LINE
                    try:
                        ## ADD DATA IF WE HAVE REACHED END
                        if ( ('[' in clean_itp[currentIndexCheck]) == False and '' != clean_itp[currentIndexCheck] ):
                            DataOfInterest.append(clean_itp[currentIndexCheck])
                        ## CASE WHERE WE HAVE NO MORE LINES LEFT!
                    except Exception:
                        pass
                    ## FILE HAS ENDED, TURNING WHILE LOOP OFF
                    break                        
                
            # Removing any blanks and then splitting to columns
            DataOfInterest = [ currentLine.split() for currentLine in DataOfInterest if len(currentLine) != 0 ]
            
            return DataOfInterest 
            
        ### FUNCTION TO EXTRACT INFORMATION FOR A GIVEN INDEX
        def extract_data_given_index(no_comment_data, index):
            '''
            This function extracts all the data given an index. It will loop through and see if the file ends
            INPUTS:
                no_comment_data: [list] data that is in a form of a list of strings
                index: [int] index where you start getting the data
            OUTPUTS:
                data: [list] list of data values which is split (no blanks!)
            '''
            # Defining the data we are interested in
            DataOfInterest = []
            currentIndexCheck = index+1 
            
            # Using while loop to go through the list to see when this thing ends
            while ('[' in no_comment_data[currentIndexCheck]) is False and currentIndexCheck != len(no_comment_data) - 1: # Checks if the file ended
                # Appending to the data set
                DataOfInterest.append(no_comment_data[currentIndexCheck])
                # Incrementing the index
                currentIndexCheck+=1     
                print(currentIndexCheck)
                print(len(no_comment_data)-1)
                
            # Removing any blanks and then splitting to columns
            DataOfInterest = [ currentLine.split() for currentLine in DataOfInterest if len(currentLine) != 0 ]
            
            return DataOfInterest
        
        ### MAIN SCRIPT
        # Finding all types that you can extract
        allTypes = [ eachLine for eachLine in self.clean_itp if '[' in eachLine ]
        
        ## FINDING IF THERE IS DUPLICATES
        duplicate_types=list(set([x for x in allTypes if allTypes.count(x) > 1]))
        if len(duplicate_types) > 0:
            print("DUPLICATION OF ITP TYPE FOUND FOR: %s"%(', '.join(duplicate_types)) )
        
        # Finding all data
        data_by_type = [ extractDataType( clean_itp=self.clean_itp, desired_type=eachType) for eachType in allTypes ]
        
        # Creating dictionary for each one
        self.itp_dict = {}
        for currentIndex in range(len(allTypes)):
            self.itp_dict[allTypes[currentIndex]] = data_by_type[currentIndex]
            
        ## LOOPING THROUGH EACH DUPLICATED TYPE
        self.itp_dict['duplicate'] = {}
        if len(duplicate_types) > 0:
            print("CREATING DUPLICATION LIST FOR ITP FILE")
            for each_duplicated_item in duplicate_types:
                ## FINDING ALL INDEXES OF THAT DUPLICATED (REMOVING THE FIRST INDEX, SINCE IT IS ALREADY TAKEN)
                duplicated_index = [ index for index, each_line in enumerate(self.clean_itp) if each_duplicated_item in each_line ][1:]
                ## LOOPING THROUGH EACH INDEX
                for each_index in duplicated_index:
                    ## GETTING DATA OF THE NEW DUPLICATION
                    data_of_duplicated = extract_data_given_index(self.clean_itp, each_index)
                    ## STORING EACH DUPLICATED VALUE
                    try:
                        self.itp_dict['duplicate'][each_duplicated_item].extend(data_of_duplicated)
                    except:
                        self.itp_dict['duplicate'][each_duplicated_item]=data_of_duplicated
    
    ### FUNCTION TO READ BINDING INFORMATION FROM ITP
    def extract_bonding( self ):
        '''
        The purpose of this script is to look into your itp dictionary, then find all the atom names, and correlate them to bonding information
        INPUTS:
            self.itp_dict: dictionary containing your different itp types
        OUTPUS:
            self.bonds: List of bonds
        '''
        # Defining the itp types you want
        atom_itp_name='[ atoms ]'    
        bond_itp_name = '[ bonds ]'
                          
        # Defining the data
        atom_itp_data = self.itp_dict[atom_itp_name]
        bond_itp_data = self.itp_dict[bond_itp_name]
        print("There are %s bonds and %s number of atoms"%(len(bond_itp_data), len(atom_itp_data)))
        
        # Getting atom numbers and atom names
        atom_num_names = [ [currentLine[0], currentLine[4]] for currentLine in atom_itp_data ]
        
        # Defining empty bonding data
        self.bonds_atomname=[]
        
        # Now, getting bonding data
        for currentBond in range(len(bond_itp_data)):
            
            # Defining atom numbers
            atom_A = bond_itp_data[currentBond][0]
            atom_B = bond_itp_data[currentBond][1]
            
            # Finding atom names
            atom_names = [ currentAtom[1] for currentAtom in atom_num_names if atom_A in currentAtom or atom_B in currentAtom ]
            
            # Appending the data
            self.bonds_atomname.append(atom_names)        
            
    ### ACTIVE FUNCTION: FINDS ATOMS BONDED GIVEN THE ATOM INDEX
    def find_atoms_bonded(self, atom_index):
        '''
        The purpose of this function is to find all the atoms bonded to another atom
        INPUTS:
            atom_index: index of the atom based on itp atom name
        OUTPUTS:
            bonded_atom_index: atom index of all the atoms bonded
        '''
        ## CREATING EMPTY LIST
        bonded_atom_index = []
        ## USING BONDING INFORMATION 
        for each_bond in self.bonds:
            ## SEEING IF ATOM INDEX WITHIN THE BOND
            if np.isin(atom_index, each_bond):
                ## FINDING ATOM INDEX THAT IS NOT THE ATOM
                atom_bonded = each_bond[np.where(each_bond != atom_index)][0]
                ## APPENDING
                bonded_atom_index.append(atom_bonded)
        return bonded_atom_index
        
    ### ACTIVE FUNCTION: PRINTS ITP FILE AS IS
    def print_itp_file(self, output_itp, output_folder ):
        '''
        The purpose of this function is to print the itp file as we currently have it
        INPUTS:
            output_itp: [string] output itp name
            output_folder: [string] output folder
        OUTPUTS:
            itp file within the output folder            
        '''
        print("\n~~~ print_itp_file from extract_itp class ~~~")
        ## DEFINING PATH
        output_itp_path = output_folder + '/' + output_itp
        
        ## CREATING THE FILE
        print("OUTPUTTING ITP FILE: %s"%(output_itp_path))  # Let user know what is being outputted
        outputfile = open(output_itp_path, "w") # Now, writing in output file
        ### WRITING HEADER
        outputfile.write("; Created by extract_itp from MDBuilders\n")
        #######################
        ### MANDATORY ITEMS ###
        #######################
        ### START BY GETTING ALL MANDATORY ITEMS ALL NONBONDED PARMS, ETC.
        for each_mandatory_item in self.mandatory_items:
            ## WRITING HEADER
            outputfile.write( "%s\n"%( each_mandatory_item ) )
            ## WRITING COMMENTS
            write_itp_type_comment(outputfile,each_mandatory_item)
        #####################
        ### MOLECULE TYPE ###
        #####################
        outputfile.write("[ moleculetype ]\n")
        write_itp_type_comment(outputfile,'[ moleculetype ]')
        outputfile.write("%s    3\n\n"%(self.residue_name))
        #############
        ### ATOMS ###
        #############
        outputfile.write("[ atoms ]\n")
        write_itp_type_comment(outputfile,'[ atoms ]')
        ### LOOPING THROUGH EACH ATOM
        for index, atom_num in enumerate(self.atom_num):
            ## PRINTING INFORMATION TO ITP FILE GIVEN THE INDEX
            outputfile.write("%4d %10s  %d %s    %s %4d  %.3f   %.4f\n"%(index+1, self.atom_type[index],
                                                                          1, self.residue_name.rjust(10, ' '), 
                                                                          self.atom_atomname[index].rjust(5, ' '), 1, 
                                                                          float(self.atom_charge[index]), 
                                                                          float(self.atom_mass[index])) )
        ##########################################
        ### BONDS / ANGLES / DIHEDRALS / PAIRS ###
        ##########################################
        
        ## TYPES TO INCLUDE
        Additional_types = ['[ bonds ]', '[ angles ]', '[ dihedrals ]', '[ pairs ]']
        for current_type in Additional_types:
            ## STRING TO OUTPUT
            type_string = add_before_and_after_2_string(current_type, '\n')
            outputfile.write(type_string)
            ## ADDING COMMENT
            write_itp_type_comment(outputfile,current_type)
            
            ### --- BONDS --- ###
            if current_type == '[ bonds ]':
                ## DEFINING BONDING INFORMATION
                bonding_info = self.bonds
                ## SEEING IF THERE ARE BONDS
                if len(bonding_info) > 0:
                    ## LOOPING EACH BOND
                    for each_bond in range(len(bonding_info)):
                        current_bond = bonding_info[each_bond]
                        current_bond_func = self.bonds_func[each_bond]                        
                        outputfile.write("%d   %d   %d\n"%(current_bond[0], current_bond[1], current_bond_func))
                        
            ### --- ANGLES --- ###
            elif current_type == '[ angles ]':
                ## DEFINING ANGLE INFORMATION
                angle_info = self.angles
                ## SETTING IF THERE ARE ANGLES
                if len(angle_info) > 0:
                    for each_angle in range(len(angle_info)):
                        current_angle = angle_info[each_angle]
                        current_angle_func =  self.angles_func[each_angle]    
                        outputfile.write("%d   %d   %d   %d\n"%(current_angle[0],
                                                                current_angle[1],
                                                                current_angle[2],
                                                                current_angle_func))
            ### --- DIHEDRALS --- ###
            elif current_type == '[ dihedrals ]':
                ## DEFINING DIHEDRAL INFORMATION
                dihedral_info = self.dihedrals
                ## SEEING IF THERE IS DIHEDRALS
                if len(dihedral_info) > 0:
                    for each_dihedral in range(len(dihedral_info)):
                        current_dihedral = dihedral_info[each_dihedral]
                        current_dihedral_func = self.dihedrals_func[each_dihedral]
                        outputfile.write("%d   %d   %d   %d    %d\n"%(current_dihedral[0],
                                                                      current_dihedral[1],
                                                                      current_dihedral[2],
                                                                      current_dihedral[3],
                                                                      current_dihedral_func))
                        
            ### --- PAIRS --- ###
            elif current_type == '[ pairs ]':
                ## DEFINING DIHEDRAL INFORMATION
                pair_info = self.pairs
                ## SEEING IF THERE IS DIHEDRALS
                if len(pair_info) > 0:
                    for each_pair in range(len(pair_info)):
                        current_pair = pair_info[each_pair]  # Correcting for residue index
                        current_pair_func = self.pairs_func[each_pair]
                        outputfile.write("%d   %d   %d\n"%(current_pair[0],
                                                          current_pair[1],
                                                          current_pair_func,
                                                          ))
        ## SUMMARY
        print("MANDATORY CATEGORIES ADDED: %s"%(' '.join(self.mandatory_items)))
        self.print_summary()
    

            
        return
        
    ### FUNCTION TO PRINT THE SUMMARY
    def print_summary(self):
        ''' This function prints a summary '''
        ## SUMMARY
        print("RESIDUE NAME: %s" %(self.residue_name))
        print("NUMBER OF MANDATORY TYPES: %s"%(self.total_mandatory))
        if self.total_mandatory != 0:
            print("TYPES ARE: %s"%(', '.join(self.matching_mandatory)))
        
        print("\n------ MOLECULAR INFORMATION -----")
        print("NUMBER OF ATOMS: %s"%(len(self.atom_num)))
        print("NUMBER OF BONDS: %s"%(len(self.bonds)))
        print("NUMBER OF PAIRS: %s"%(len(self.pairs)))
        print("NUMBER OF ANGLES: %s"%(len(self.angles)))
        print("NUMBER OF DIHEDRALS: %s"%(len(self.dihedrals)))
        return
        
        
    ### MAIN FUNCTION
    def __init__(self, itp_file_path):
        ## PRINTING
        print("\n--- CLASS: extract_itp ---")
        print("\n*** EXTRACTING ITP FILE FROM: %s ***"%(itp_file_path))
        ## DEFINING ITP FILE NAME
        self.itp_file_path = itp_file_path            
        ## READ ITP FILE
        self.readITP()
        ## CONVERTING ITP FILE TO DICTIONARY
        self.itp2dictionary()
        
        ## FINDING RESIDUE NAME
        self.residue_name = self.itp_dict['[ moleculetype ]'][0][0]
        
        ## DEFINING MANDATORY ITP DETAILS
        mandatory_itp_list = [ 
                                '[ nonbond_params ]', # LJ parameters
                                '[ atomtypes ]', # Atom types 
                                '[ bondtypes ]', # Bond types
                                '[ pairtypes ]', # Pair types
                                '[ dihedraltypes ]', # Dihedral types
                                '[ constrainttypes ]', # Constraint types
                              ]
        
        ## FINDING IF ANY OF THE MANDATORY ITP DETAILS ARE LISTED
        self.matching_mandatory = list(set(mandatory_itp_list) & set(self.itp_dict.keys()))
        self.total_mandatory = len(self.matching_mandatory)
        
        ## FINDING ATOMTYPES / NONBONDED INFORMATION
        self.mandatory_items = []
        if self.total_mandatory != 0:
            for each_mandatory_index in range(self.total_mandatory):
                self.mandatory_items.append( [self.matching_mandatory[each_mandatory_index],  self.itp_dict[self.matching_mandatory[each_mandatory_index]] ] )
        
        ## FINDING ATOM INFORMATION
        if '[ atoms ]' in self.itp_dict.keys():
            self.atom_num = [ int(atom[0]) for atom in self.itp_dict['[ atoms ]']]
            self.atom_type = [ atom[1] for atom in self.itp_dict['[ atoms ]']]
            self.atom_resnr = [ atom[2] for atom in self.itp_dict['[ atoms ]']]
            self.atom_atomname = [ atom[4] for atom in self.itp_dict['[ atoms ]']]
            self.atom_charge = [ float(atom[6]) for atom in self.itp_dict['[ atoms ]']]
            self.atom_mass = [  float(atom[7]) for atom in self.itp_dict['[ atoms ]']]
        else:
            self.atom_num, self.atom_type, self.atom_resnr, self.atom_atomname, self.atom_charge, self.atom_mass = [], [], [], [], [], []
        
        ## FINDING BONDING DETAILS
        ## FINDING BINDING INFORMATION (only if we have bonds to begin with!)
        if '[ bonds ]' in self.itp_dict.keys():
            self.bonds = np.array( [ bonds[0:2] for bonds in self.itp_dict['[ bonds ]']] ).astype('int')
            try:
                self.bonds_func = np.array( [ bonds[-1] for bonds in self.itp_dict['[ bonds ]']] ).astype('int')
            except: # Bond function typically the third column
                self.bonds_func = np.array( [ bonds[2] for bonds in self.itp_dict['[ bonds ]']] ).astype('int')
            self.extract_bonding()
        else:
            self.bonds, self.bonds_func, self.bonds_atomname = [], [], []
            
        ## FINDING PAIRS INFORMATION                
        if '[ pairs ]' in self.itp_dict.keys():
            self.pairs = np.array( [ pairs[0:2] for pairs in self.itp_dict['[ pairs ]']] ).astype('int')
            self.pairs_func = np.array( [ pairs[2] for pairs in self.itp_dict['[ pairs ]']] ).astype('int')
        else:
            self.pairs, self.pairs_func = [], []
            
        ## FINDING ANGLE DETAILS
        if '[ angles ]' in self.itp_dict.keys():
            self.angles = np.array( [ angle[0:3] for angle in self.itp_dict['[ angles ]']] ).astype('int')
            self.angles_func = np.array( [ angle[3] for angle in self.itp_dict['[ angles ]']] ).astype('int')
        else:
            self.angles, self.angles_func = [], []
        
        ## FINDING DIHEDRALS
        if '[ dihedrals ]' in self.itp_dict.keys():
            self.dihedrals = np.array( [ dihedral[0:4] for dihedral in self.itp_dict['[ dihedrals ]']] ).astype('int')
            self.dihedrals_func = np.array( [ dihedral[4] for dihedral in self.itp_dict['[ dihedrals ]']] ).astype('int')
        else:
            self.dihedrals, self.dihedrals_func = [], []
        
        ## PRINTING SUMMARY
        self.print_summary()        
        
        ## AT THE END, ADD ANY DUPLICATES
        if len(self.itp_dict['duplicate']) > 0:
            print("DUPLICATES FOUND -- ADDING THEM TO ITP FILE")
            duplicate_dict = self.itp_dict['duplicate']
            ## LOOPING THROUGH KEYS
            for each_key in duplicate_dict.keys():
                ## FINDING WHICH ONE WE HAVE
                if '[ dihedrals ]' == each_key:
                    ## APPENDING TO DATA
                    self.dihedrals = np.concatenate( (self.dihedrals, np.array( [ dihedral[0:4] for dihedral in duplicate_dict[each_key] ] ).astype('int') )  )
                    self.dihedrals_func = np.concatenate( (self.dihedrals_func, np.array( [ dihedral[4] for dihedral in duplicate_dict[each_key] ] ).astype('int') )  )
                else:
                    print("UNIDENTIFIED DUPLICATE FOR KEY: %s"%(each_key) )
                    
########################################
### CLASS FUNCTION TO READ XVG FILES ###
########################################
class read_xvg:
    '''
    The purpose of this function is to read xvg files outputted by GROMACS
    INPUTS:
        xvg_file: [str] full path to the xvg file
    OUTPUTS:
        self.xvg_lines: [list] each line of the str file
        self.xvg_data: [list] xvg data splitted as a list of lists
    FUNCTIONS:
        read_xvg_lines: reads the xvg file
        find_xvg_data: finds the data within the xvg file
        print_summary: prints the summary
    ALGORITHM:
        - Load in the xvg file
        - Extract the point when the data starts
        - Store the data as a variable
        - POST-CLASS: Use this class to analyze the xvg file
    '''
    ### INITIALIZING
    def __init__(self, xvg_file):
        ## STORING INPUTS
        self.xvg_file = xvg_file
        
        ## READING XVG LINES
        self.read_xvg_lines()
        
        ## FINDING THE DATA
        self.find_xvg_data()
        
        ## PRINTING SUMMARY
        self.print_summary()
        return
    ### FUNCTION TO PRINT SUMMARY
    def print_summary(self):
        """ This function prints the summary """
        print("XVG File loaded: %s"%(self.xvg_file))
        return
    
    ### FUNCTION TO OPEN AND READ XVG FILE
    def read_xvg_lines(self):
        '''
        The purpose of this function is to read the xvg file
        INPUTS:
            self: class property
        OUTPUTS:
            self.xvg_lines: [list] each line of the str file
        '''
        with open(self.xvg_file, 'r') as data_file:
            self.xvg_lines = data_file.readlines()
        return
    
    ### FUNCTION TO FIND THE DATA
    def find_xvg_data(self):
        '''
        The purpose of this function is to find the data within the xvg lines
        INPUTS:
            self: class object
        OUTPUTS:
            self.xvg_data: [list] xvg data splitted as a list of lists
        '''
        ## FINDING INDEX WHERE THE LAST @ IS IN (Signifies the end of the legend section)
        index = [i for i, j in enumerate(self.xvg_lines) if '@' in j][-1]
        ## FINDING THE DATA
        self.xvg_data= [ line.split() for line in self.xvg_lines[index+1:]]
    
    
### FUNCTION TO READ ITP FILE
def load_itp_file( itp_file_path, itp_file_name ='match', residue_names = None,):
    '''
    The purpose of this function is to find the itp file and load it based on a residue name
    INPUTS:
        itp_file_path: [str]
            itp file path to check for
        itp_file_name: [str, default='match']
            file name, if 'match', then we will try to match with residue names
        residue_names: [str, default=None]
            residue name that you are interested in
    OUTPUTS:
        itp_file: [class]
            itp file that has been extracted
    '''
    if itp_file_name == 'match':
        print("Since itp file was set to 'match', we are going to try to find an ITP file that matches your ligand!")
        ## USING GLOB TO FIND ALL ITP FILES
        itp_files = glob.glob( itp_file_path + '/*.itp' )
        for full_itp_path in itp_files:
            print("Checking itp path: %s"%(full_itp_path) )
            try:
                itp_info = extract_itp(full_itp_path)
                ## STORE ONLY IF THE RESIDUE NAME MATCHES THE LIGAND
                if itp_info.residue_name == residue_names:
                    itp_file = itp_info
                    break ## Breaking out of for loop
            except Exception: # <-- if error in reading the itp file (happens sometimes!)
                pass
    else:
        itp_file = extract_itp(itp_file_path + '/' + itp_file_name)
    ## SEEING IF ITP FILE EXISTS
    try:
        print("Found itp file! ITP file path: %s"%(itp_file.itp_file_path))
    except NameError:
        print("Error! No ITP file for names found: %s"%(residue_names) )
        print("Perhaps, check your input files and make sure an itp file with the ligand name residue is there!")
        print("Stopping here to prevent subsequent errors!")
        sys.exit()
    return itp_file

### FUNCTION TO GET ALL DIHEDRAL ANGLES IN A FORM OF ATOM INDICES
def convert_dihedrals_to_atom_index( atom_index, itp_file ):
    '''
    The purpose of this function is to convert the atom index from md.traj and dihedrals from itp_file to atom numbers with respect to the md.traj
    INPUTS:
        atom_index: [np.array, shape=(N,1)]
            atom index with taken from md.traj
                e.g. array([7200, 7201, 7202,
        itp_file: [object]
            itp file object taken from loading itp file script
    OUTPUTS:
        dihedrals_atom_index: [np.array, shape=(num_dihedrals, 4)]
            dihedrals that are based on the atom index of your trajectory
            e.g.:
                array([[7208, 7200, 7201, 7202],
               [7208, 7200, 7201, 7203],
               [7208, 7200, 7201, 7205], ...
    '''
    ## CONVERTING DIHEDRALS FROM ITP FILE TO INDEXING OF MD TRAJ
    dihedrals_renumbered = itp_file.dihedrals - 1
    ## USING ATOM INDEX TO GET DIHEDRAL ANGLE INDEX
    dihedrals_atom_index = atom_index[dihedrals_renumbered]
    return dihedrals_atom_index

### FUNCTION TO CONVERT ANY SIZE NUMPY ARRAY TO ATOM SYMBOLS
def convert_atom_index_to_elements( traj, atom_index, element_type="symbol" ):
    '''
    The purpose of this function is to convert any size atom index to symbols. This is highly useful for cases when you want to find a specific bond, for example.
    INPUTS:
        traj: [md.traj]
            trajectory from md.traj
        atom_index: [np.array, any shape]
            numpy array taking any shape. The atom index should correspond with the indices found in traj.
            e.g.:
                array([[7208, 7200, 7201, 7202],
                       [7208, 7200, 7201, 7203],
                       [7208, 7200, 7201, 7205],...)
        element_type: [str]
            element type that you want your output to be
                symbol: if you only want symbols (e.g. 'O')
                name: if you want name of the atom (e.g. "C1")
    OUTPUTS:
        atom_symbols: [np.array, same shape as atom_index]
            atom symbols corresponding to the atom index
            e.g.:
                array([['H', 'C', 'C', 'H'],
                       ['H', 'C', 'C', 'C'],
                       ['H', 'C', 'C', 'O'],...)
    '''
    ## START BY FINDING THE SHAPE AND SIZE
    shape = atom_index.shape
    ## FLATTENING ATOM INDEX ARRAY
    flatten_atom_index = atom_index.flatten()
    ## USING THE ATOM INDEX TO FIND ALL ATOMIC SYMBOLS
    if element_type == "symbol":
        flatten_array = np.array([ traj.topology.atom(each_atom).element.symbol for each_atom in flatten_atom_index ])
    elif element_type == "name":
        flatten_array = np.array([ traj.topology.atom(each_atom).name for each_atom in flatten_atom_index ])
    else:
        print("Error! Element type '%s' is not found! Please check to ensure element type is available for conerting atom index to elements"%(element_type) )
    ## NOW, USING THE NEW SYMBOLS AND RESHAPING
    atom_elements = flatten_array.reshape(shape)
    return atom_elements