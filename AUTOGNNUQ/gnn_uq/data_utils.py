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
PATH_DISTANCE_BINS = list(range(10))
THREE_D_DISTANCE_MAX = 20
THREE_D_DISTANCE_STEP = 1
THREE_D_DISTANCE_BINS = list(
    range(0, THREE_D_DISTANCE_MAX + 1, THREE_D_DISTANCE_STEP))

# len(choices) + 1 to include room for uncommon values; + 2 at end for IsAromatic and mass
ATOM_FDIM = sum(len(choices) + 1 for choices in ATOM_FEATURES.values()) + 2
BOND_FDIM = 14

# Memoization
SMILES_TO_GRAPH = {}


def onek_encoding_unk(value, choices):
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1

    return encoding


def atom_featurizer(atom):
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


def get_data(path, max_data_size=None):

    max_data_size = max_data_size or float('inf')

    # if already exists
    pickle_path = path.split('csv')[0] + 'pickle'

    if os.path.exists(pickle_path):
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
    else:
        # load data
        with open(path, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # skip header

            data = {'smiles': [],
                    'atom_features': [],
                    'bond_features': [],
                    'pair_indices': [],
                    'label': []}

            for line in reader:
                smiles = line[0]
                mol = Chem.MolFromSmiles(smiles, sanitize=True)
                atom_feature, bond_feature, pair_index = mol_to_graph(mol)
                data['smiles'].append(smiles)
                data['atom_features'].append(atom_feature)
                data['bond_features'].append(bond_feature)
                data['pair_indices'].append(pair_index)
                data['label'].append(float(line[1]))

                if len(data) >= max_data_size:
                    break

        data = format_data(data)

        with open(pickle_path, 'wb') as f:
            pickle.dump(data, f)

    return data


def format_data(data):
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
    mol = Chem.MolFromSmiles(mol) if type(mol) == str else mol
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        mol=mol, includeChirality=False)
    return scaffold


def scaffold_to_smiles(mols, use_indices=True):
    scaffolds = defaultdict(set)

    for i, mol in enumerate(mols):
        scaffold = generate_scaffold(mol)
        if use_indices:
            scaffolds[scaffold].add(i)
        else:
            scaffolds[scaffold].add(mol)

    return scaffolds