"""
Adapted from Chemprop.data.utils.py

Author: Shengli Jiang
Email: sjiang87@wisc.edu
"""
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict

import os
import csv
import pickle
import numpy as np

# Atom feature sizes
MAX_ATOMIC_NUM = 100
ATOM_FEATURES = {
    'atomic_num': list(range(MAX_ATOMIC_NUM)),
    'degree': [0, 1, 2, 3, 4, 5],
    'formal_charge': [-1, -2, 1, 2, 0],
    'chiral_tag': [0, 1, 2, 3],
    'num_Hs': [0, 1, 2, 3, 4],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ],
}

# Distance feature sizes
PATH_DISTANCE_BINS    = list(range(10))
THREE_D_DISTANCE_MAX  = 20
THREE_D_DISTANCE_STEP = 1
THREE_D_DISTANCE_BINS = list(range(0, THREE_D_DISTANCE_MAX + 1, THREE_D_DISTANCE_STEP))

# len(choices) + 1 to include room for uncommon values; + 2 at end for IsAromatic and mass
ATOM_FDIM = sum(len(choices) + 1 for choices in ATOM_FEATURES.values()) + 2
BOND_FDIM = 14

# Memoization
SMILES_TO_GRAPH = {}

def onek_encoding_unk(value, choices):
    """
    Encode a categorical value using one-hot encoding, handling unknown values.

    Args:
        value: The categorical value to encode.
        choices (list): List of possible categorical choices.

    Returns:
        list: The one-hot encoding of the value.

    """
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1
    return encoding


def atom_featurizer(atom):
    """
    Generate features for a given atom.

    Args:
        atom (rdkit.Chem.Atom): The RDKit Atom object.

    Returns:
        list: List of features representing the atom.

    """
    features = onek_encoding_unk(atom.GetAtomicNum() - 1, ATOM_FEATURES['atomic_num']) + \
        onek_encoding_unk(atom.GetTotalDegree(), ATOM_FEATURES['degree']) + \
        onek_encoding_unk(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge']) + \
        onek_encoding_unk(int(atom.GetChiralTag()), ATOM_FEATURES['chiral_tag']) + \
        onek_encoding_unk(int(atom.GetTotalNumHs()), ATOM_FEATURES['num_Hs']) + \
        onek_encoding_unk(int(atom.GetHybridization()), ATOM_FEATURES['hybridization']) + \
        [1 if atom.GetIsAromatic() else 0] + \
        [atom.GetMass() * 0.01]  # scaled to about the same range as other features
    return features


def bond_featurizer(bond):
    """
    Generate features for a given bond.

    Args:
        bond (rdkit.Chem.Bond or None): The RDKit Bond object or None if there is no bond.

    Returns:
        list: List of features representing the bond.

    """
    if bond is None:
        fbond = [1] + [0] * (BOND_FDIM - 1)
    else:
        bt = bond.GetBondType()
        fbond = [
            0,  # bond is not None
            bt == Chem.rdchem.BondType.SINGLE,
            bt == Chem.rdchem.BondType.DOUBLE,
            bt == Chem.rdchem.BondType.TRIPLE,
            bt == Chem.rdchem.BondType.AROMATIC,
            (bond.GetIsConjugated() if bt is not None else 0),
            (bond.IsInRing() if bt is not None else 0)
        ]
        fbond += onek_encoding_unk(int(bond.GetStereo()), list(range(6)))
    return fbond


def mol_to_graph(mol):
    """
    Convert an RDKit molecule into graph representation.

    Args:
        mol (rdkit.Chem.Mol): The RDKit molecule object.

    Returns:
        tuple: Tuple containing atom features, bond features, and pair indices.

    """
    # Initialize graph
    atom_features = []
    bond_features = []
    pair_indices = []

    for atom in mol.GetAtoms():
        atom_features.append(atom_featurizer(atom))

        # Add self-loops
        pair_indices.append([atom.GetIdx(), atom.GetIdx()])
        bond_features.append(bond_featurizer(None))

        for neighbor in atom.GetNeighbors():
            bond = mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx())
            pair_indices.append([atom.GetIdx(), neighbor.GetIdx()])
            bond_features.append(bond_featurizer(bond))

    return np.array(atom_features), np.array(bond_features), np.array(pair_indices)


def get_data(path, tasks, max_data_size=None):
    """
    Load and preprocess data from a CSV file containing molecular information.

    Args:
        path (str): Path to the CSV file.
        tasks (list): List of strings specifying the tasks to extract from the CSV file.
        max_data_size (int, optional): Maximum number of data samples to load. Defaults to None.

    Returns:
        dict: Dictionary containing the preprocessed data.
    """

    max_data_size = max_data_size or float('inf')

    # if already exists
    pickle_path = path.split('csv')[0] + 'pickle'

    if os.path.exists(pickle_path) :
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
    else:
        # load data
        with open(path, 'r') as f:
            header = next(csv.reader(f))
        
        smile_idx = header.index("smiles")
        label_idx = [header.index(task) for task in tasks if task in header]
            
        with open(path, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # skip header

            data = {'smiles': [],
                    'atom_features': [],
                    'bond_features': [],
                    'pair_indices': [],
                    'label': []}

            for line in reader:
                smiles = line[smile_idx]
                mol = Chem.MolFromSmiles(smiles, sanitize=True)
                atom_feature, bond_feature, pair_index = mol_to_graph(mol)
                data['smiles'].append(smiles)
                data['atom_features'].append(atom_feature)
                data['bond_features'].append(bond_feature)
                data['pair_indices'].append(pair_index)
                data['label'].append([float(line[i]) for i in label_idx])

                if len(data) >= max_data_size:
                    break

        data = format_data(data)

        with open(pickle_path, 'wb') as f:
            pickle.dump(data, f)

    return data


def format_data(data):
    """
    Format the raw molecular data into a standard format suitable for further processing.

    Args:
        data (dict): Dictionary containing the raw molecular data.

    Returns:
        dict: Dictionary containing the formatted molecular data.

    """
    num_mol = len(data['atom_features'])
    num_atom = [data['atom_features'][i].shape[0] for i in range(num_mol)]
    num_bond = [data['bond_features'][i].shape[0] for i in range(num_mol)]
    max_atom = np.max(num_atom) + 1
    max_bond = np.max(num_bond) + max_atom

    atom_features_ = np.zeros((num_mol, max_atom, ATOM_FDIM))
    bond_features_ = np.zeros((num_mol, max_bond, BOND_FDIM))
    pair_indices_ = np.ones((num_mol, max_bond, 2)) * (max_atom - 1)
    masks_ = np.zeros((num_mol, max_atom))

    for i in range(num_mol):
        atom_features_[i][:num_atom[i]] = data['atom_features'][i]
        bond_features_[i][:num_bond[i]] = data['bond_features'][i]
        pair_indices_[i][:num_bond[i]] = data['pair_indices'][i]
        non_node_pair = np.arange(num_atom[i], max_atom)[..., None]
        fill_pair = np.repeat(non_node_pair, repeats=2, axis=1)
        pair_indices_[i][num_bond[i]:num_bond[i]+max_atom-num_atom[i]] = fill_pair
        masks_[i][:num_atom[i]] = np.ones((num_atom[i], ))

    data['atom_features'] = atom_features_
    data['bond_features'] = bond_features_
    data['pair_indices'] = pair_indices_
    data['masks'] = masks_
    data['label'] = np.array(data['label'])
    
    return data


def split_data(data, split_type='random', sizes=(0.5, 0.2, 0.3), seed=0, show_mol=False):
    """
    Split the molecular data into training, validation, and test sets according to the specified split type.

    Args:
        data (dict): Dictionary containing the molecular data.
        split_type (str, optional): Type of data split. Defaults to 'random'.
        sizes (tuple, optional): Sizes of the training, validation, and test sets. Defaults to (0.5, 0.2, 0.3).
        seed (int, optional): Seed for random number generation. Defaults to 0.
        show_mol (bool, optional): Whether to return molecular structures along with data splits. Defaults to False.

    Returns:
        list: Depending on 'show_mol', returns either a list containing the molecular structures and labels 
        for each split, or lists containing the feature matrices and labels for each split.

    """

    if split_type == 'random':
        num_mol = len(data['atom_features'])
        train_size = int(num_mol * sizes[0])
        train_val_size = int(num_mol * (1 - sizes[-1]))

        atom_features_ = np.random.RandomState(
            seed).permutation(data['atom_features'])
        bond_features_ = np.random.RandomState(
            seed).permutation(data['bond_features'])
        pair_indices_ = np.random.RandomState(
            seed).permutation(data['pair_indices'])
        masks_ = np.random.RandomState(
            seed).permutation(data['masks'])
        label_ = np.random.RandomState(seed).permutation(data['label'])

        x_train = [atom_features_[:train_size],
                   pair_indices_[:train_size],
                   bond_features_[:train_size],
                   masks_[:train_size]]

        x_valid = [atom_features_[train_size:train_val_size],
                   pair_indices_[train_size:train_val_size],
                   bond_features_[train_size:train_val_size],
                   masks_[train_size:train_val_size]]

        x_test = [atom_features_[train_val_size:],
                  pair_indices_[train_val_size:],
                  bond_features_[train_val_size:],
                  masks_[train_val_size:]]

        y_train = label_[:train_size]

        y_valid = label_[train_size:train_val_size]

        y_test = label_[train_val_size:]
        
        mol_ = np.random.RandomState(
            seed).permutation(np.array(data['smiles']))
        mol_train = mol_[:train_size]
        mol_valid = mol_[train_size:train_val_size]
        mol_test = mol_[train_val_size:]

    elif split_type == 'scaffold':
        if show_mol:
            mol_train, y_train, mol_valid, y_valid, mol_test, y_test = scaffold_split(
                data, sizes, show_mol)
        else:
            x_train, y_train, x_valid, y_valid, x_test, y_test = scaffold_split(
                data, sizes, show_mol)
            
    if show_mol:
        return mol_train, y_train, mol_valid, y_valid, mol_test, y_test
    else:
        return x_train, y_train, x_valid, y_valid, x_test, y_test


def scaffold_split(data, sizes=(0.5, 0.2, 0.3), show_mol=False):
    """
    Perform scaffold-based splitting of the molecular data into training, validation, and test sets.

    Args:
        data (dict): Dictionary containing the molecular data.
        sizes (tuple, optional): Sizes of the training, validation, and test sets. Defaults to (0.5, 0.2, 0.3).
        show_mol (bool, optional): Whether to return molecular structures along with data splits. Defaults to False.

    Returns:
        list: Depending on 'show_mol', returns either a list containing the molecular structures and labels 
        for each split, or lists containing the feature matrices and labels for each split.

    """
    # Split
    num_mol = len(data['atom_features'])
    train_size, val_size = sizes[0] * num_mol, sizes[1] * num_mol
    train, val, test = [], [], []
    train_scaffold_count, val_scaffold_count, test_scaffold_count = 0, 0, 0

    # Map from scaffold to index in the data
    scaffold_to_indices = scaffold_to_smiles(data['smiles'], use_indices=True)
    index_sets = sorted(list(scaffold_to_indices.values()),
                        key=lambda index_set: len(index_set),
                        reverse=True)

    for index_set in index_sets:
        if len(train) + len(index_set) <= train_size:
            train += index_set
            train_scaffold_count += 1
        elif len(val) + len(index_set) <= val_size:
            val += index_set
            val_scaffold_count += 1
        else:
            test += index_set
            test_scaffold_count += 1

    atom_features_ = data['atom_features']
    bond_features_ = data['bond_features']
    pair_indices_ = data['pair_indices']
    masks_ = data['masks']
    label_ = data['label']

    x_train = [atom_features_[train],
               pair_indices_[train],
               bond_features_[train],
               masks_[train]]

    x_valid = [atom_features_[val],
               pair_indices_[val],
               bond_features_[val],
               masks_[val]]

    x_test = [atom_features_[test],
              pair_indices_[test],
              bond_features_[test],
              masks_[test]]

    y_train = label_[train]

    y_valid = label_[val]

    y_test = label_[test]
    
    mol_ = np.array(data['smiles'])
    mol_train = mol_[train]
    mol_valid = mol_[val]
    mol_test = mol_[test]
    
    if show_mol:
        return mol_train, y_train, mol_valid, y_valid, mol_test, y_test
    else:
        return x_train, y_train, x_valid, y_valid, x_test, y_test


def generate_scaffold(mol):
    """
    Generate the scaffold for a given molecule.

    Args:
        mol (str or Mol): SMILES representation of the molecule or RDKit Mol object.

    Returns:
        str: SMILES representation of the scaffold.

    """
    mol = Chem.MolFromSmiles(mol) if type(mol) == str else mol
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        mol=mol, includeChirality=False)
    return scaffold


def scaffold_to_smiles(mols, use_indices=True):
    """
    Generate a mapping from scaffold to molecule indices or molecules.

    Args:
        mols (list): List of molecules in SMILES format.
        use_indices (bool, optional): If True, maps scaffolds to molecule indices. If False, maps scaffolds to molecules themselves. Defaults to True.

    Returns:
        defaultdict: A defaultdict where keys are scaffold SMILES strings and values are sets of molecule indices or molecules.

    """
    scaffolds = defaultdict(set)

    for i, mol in enumerate(mols):
        scaffold = generate_scaffold(mol)
        if use_indices:
            scaffolds[scaffold].add(i)
        else:
            scaffolds[scaffold].add(mol)

    return scaffolds