# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 22:26:12 2021

@author: sqin34
"""

import pandas as pd
import torch
import dgl
from solvgnn.util.atom_feat_encoding import CanonicalAtomFeaturizer 
from solvgnn.util.molecular_graph import mol_to_bigraph
from rdkit import Chem,DataStructs
from rdkit.Chem.Draw import MolToFile
from rdkit.Chem import AllChem
import numpy as np
    
def biPxy(x1, solv1_gam, solv2_gam, solv1_psat, solv2_psat):
    solv1_p = x1 * np.exp(solv1_gam)*solv1_psat
    solv2_p = (1-x1) * np.exp(solv2_gam)*solv2_psat
    equi_p = solv1_p+solv2_p
    y1 = solv1_p / equi_p
    return y1, equi_p

class solvent_dataset_binary:

    def __init__(self, input_file_path = "./solvgnn/data/output_binary_all.csv",
                 solvent_list_path = "./solvgnn/data/solvent_list.csv",
                 generate_all = False,
                 return_hbond=True, 
                 return_comp=True, 
                 return_gamma=True):
        if input_file_path:
            solvent_list = pd.read_csv(solvent_list_path,index_col='solvent_id')
            self.dataset = pd.read_csv(input_file_path)
            self.solvent_names = solvent_list['solvent_name'].to_dict()
            self.solvent_smiles = solvent_list['smiles_can'].to_dict()
        self.solvent_data = {}
        self.solvent_fp = {}
        self.return_hbond = return_hbond
        self.return_comp = return_comp
        self.return_gamma = return_gamma
        if generate_all:
            self.generate_all()
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self,idx):
        sample = {}
        if torch.is_tensor(idx):
            idx = idx.tolist()
        data = self.dataset.iloc[idx]
        ids = [data['solv1'], data['solv2']]
        solv1 = self.solvent_data[ids[0]]
        solv2 = self.solvent_data[ids[1]]
        sample['g1'] = solv1[0]
        sample['g2'] = solv2[0]
        sample['intra_hb1'] = solv1[3]
        sample['intra_hb2'] = solv2[3]
        sample['inter_hb'] = min(solv1[1],solv2[2]) + min(solv1[2],solv2[1])
        sample['solv1_x'] = data['solv1_x']
        sample['gamma1'] = data['solv1_gamma']
        sample['gamma2'] = data['solv2_gamma']
        return sample
    def generate_sample(self,chemical_list,smiles_list,solv1_x,gamma_list=None):
        solvent_data = {}
        for i,sml in enumerate(smiles_list):
            solvent_data[chemical_list[i]] = []
            sml = Chem.MolToSmiles(Chem.MolFromSmiles(sml))
            mol = Chem.MolFromSmiles(sml)
            solvent_data[chemical_list[i]].append(mol_to_bigraph(mol,add_self_loop=True,
                                               node_featurizer=CanonicalAtomFeaturizer(),
                                               edge_featurizer=None,
                                               canonical_atom_order=False,
                                               explicit_hydrogens=False,
                                               num_virtual_nodes=0
                                               ))
            hba = Chem.rdMolDescriptors.CalcNumHBA(mol)
            hbd = Chem.rdMolDescriptors.CalcNumHBD(mol)
            solvent_data[chemical_list[i]].append(hba)
            solvent_data[chemical_list[i]].append(hbd)
            solvent_data[chemical_list[i]].append(min(hba,hbd))
        sample = {}
        solv1 = solvent_data[chemical_list[0]]
        solv2 = solvent_data[chemical_list[1]]
        sample['g1'] = solv1[0]
        sample['g2'] = solv2[0]
        sample['intra_hb1'] = solv1[3]
        sample['intra_hb2'] = solv2[3]
        sample['inter_hb'] = min(solv1[1],solv2[2]) + min(solv1[2],solv2[1])
        sample['solv1_x'] = solv1_x
        if gamma_list:            
            sample['gamma1'] = gamma_list[0]
            sample['gamma2'] = gamma_list[1]
        return sample
    def generate_all(self):
        for solvent_id in self.solvent_smiles:
            mol = Chem.MolFromSmiles(self.solvent_smiles[solvent_id])
            self.solvent_data[solvent_id] = []
            self.solvent_data[solvent_id].append(mol_to_bigraph(mol,add_self_loop=True,
                                                                node_featurizer=CanonicalAtomFeaturizer(),
                                                                edge_featurizer=None,
                                                                canonical_atom_order=False,
                                                                explicit_hydrogens=False,
                                                                num_virtual_nodes=0
                                                                ))
            hba = Chem.rdMolDescriptors.CalcNumHBA(mol)
            hbd = Chem.rdMolDescriptors.CalcNumHBD(mol)
            self.solvent_data[solvent_id].append(hba)
            self.solvent_data[solvent_id].append(hbd)
            self.solvent_data[solvent_id].append(min(hba,hbd))
            
    def search_chemical(self,chemical_name):
        for solvent_id in self.solvent_names:
            if chemical_name.lower() == self.solvent_names[solvent_id].lower():
                print(solvent_id, self.solvent_names[solvent_id], self.solvent_smiles[solvent_id])
                return [solvent_id,self.dataset[(self.dataset["solv1"]==solvent_id)|(self.dataset["solv2"]==solvent_id)].index.to_list()]
    def search_chemical_pair(self,chemical_list):
        solv1_match = self.search_chemical(chemical_list[0])[0]
        solv2_match = self.search_chemical(chemical_list[1])[0]
        return [[solv1_match,solv2_match],\
                 self.dataset[((self.dataset["solv1"]==solv1_match)&(self.dataset["solv2"]==solv2_match))|\
                            ((self.dataset["solv2"]==solv1_match)&(self.dataset["solv1"]==solv2_match))].index.to_list()]                 

            
    def node_to_atom_list(self, idx):
        g1 = self.__getitem__(idx)['g1']
        g2 = self.__getitem__(idx)['g2']
        allowable_set = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca',
                         'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn',
                         'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au',
                         'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb']
        node_feat1 = g1.ndata['h'].numpy()[:,0:len(allowable_set)]
        atom_list1 = []
        for i in range(g1.number_of_nodes()):
            atom_list1.append(allowable_set[np.where(node_feat1[i]==1)[0][0]])
        node_feat2 = g2.ndata['h'].numpy()[:,0:len(allowable_set)]
        atom_list2 = []
        for i in range(g2.number_of_nodes()):
            atom_list2.append(allowable_set[np.where(node_feat2[i]==1)[0][0]])
        return {'atom_list': [atom_list1, atom_list2]}

    def get_smiles(self, idx):
        return {'smiles':[self.dataset['solv1_smiles'].iloc[idx],
                          self.dataset['solv2_smiles'].iloc[idx]]}
    def get_solvx(self, idx):
        return {'solv_x':[self.dataset['solv1_x'].iloc[idx],
                           self.dataset['solv2_x'].iloc[idx]]}
    
    def generate_solvsys(self,batch_size=5):
        n_solv = 2
        solvsys = dgl.DGLGraph()
        solvsys.add_nodes(n_solv*batch_size)
        src = torch.arange(batch_size)
        dst = torch.arange(batch_size,n_solv*batch_size)
        solvsys.add_edges(torch.cat((src,dst)),torch.cat((dst,src)))
        solvsys.add_edges(torch.arange(n_solv*batch_size),torch.arange(n_solv*batch_size))    
        return solvsys

    def get_fp(self,idx=None,smiles_list=None,radius=3,fpsize=1024):
        if idx:
            smiles_list = self.get_smiles(idx)['smiles']            
        fp_mat = []
        for sml in smiles_list:
            mol = Chem.MolFromSmiles(sml)
            fp_mat.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=fpsize))
        return fp_mat

    def generate_all_fp(self, radius=3, fpsize=1024, return_full_matrix=False):
        for solvent_id in self.solvent_smiles:
            mol = Chem.MolFromSmiles(self.solvent_smiles[solvent_id])
            self.solvent_fp[solvent_id] = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=fpsize)
        if return_full_matrix:
            fp_mat_all = []
            for idx in range(len(self.dataset)):
                fp_mat_all.append(self.solvent_fp[self.dataset.iloc[idx].solv1])
                fp_mat_all.append(self.solvent_fp[self.dataset.iloc[idx].solv2])
            return fp_mat_all


def sample_ternary_x(solv1_x_range,solv2_x_range):
    x_list = []
    solv1_x_range = np.array(solv1_x_range*100).astype("int")
    solv2_x_range = np.array(solv2_x_range*100).astype("int")
    for i in solv1_x_range:
        for j in solv2_x_range:
            if i+j <= 100:
                k = 100-i-j
                x_list.append([i,j,k])
    return np.array(x_list)/100

def terPxy(solv1_x_range,solv2_x_range,solv1_gam,solv2_gam,solv3_gam,solv1_psat,solv2_psat,solv3_psat):
    solv1_p = solv1_x_range*np.exp(solv1_gam)*solv1_psat
    solv2_p = solv2_x_range*np.exp(solv2_gam)*solv2_psat
    solv3_p = (1-solv1_x_range-solv2_x_range)*np.exp(solv3_gam)*solv3_psat
    equi_p = solv1_p+solv2_p+solv3_p
    y = np.array([solv1_p/equi_p,solv2_p/equi_p,solv3_p/equi_p])
    return y,equi_p
    
class solvent_dataset_ternary:

    def __init__(self, input_file_path = "./solvgnn/data/output_ternary_all.csv",
                 solvent_list_path = "./solvgnn/data/solvent_list.csv",
                 generate_all=False,
                 return_hbond=True, 
                 return_comp=True, 
                 return_gamma=True):
        if input_file_path:
            solvent_list = pd.read_csv(solvent_list_path,index_col='solvent_id')
            self.dataset = pd.read_csv(input_file_path)
            self.solvent_names = solvent_list['solvent_name'].to_dict()
            self.solvent_smiles = solvent_list['smiles_can'].to_dict()
        self.solvent_data = {}
        self.return_hbond = return_hbond
        self.return_comp = return_comp
        self.return_gamma = return_gamma
        if generate_all:
            self.generate_all()
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self,idx):
        sample = {}
        if torch.is_tensor(idx):
            idx = idx.tolist()
        data = self.dataset.iloc[idx]
        ids = [data['solv1'], data['solv2'], data['solv3']]
        solv1 = self.solvent_data[ids[0]]
        solv2 = self.solvent_data[ids[1]]
        solv3 = self.solvent_data[ids[2]]
        sample['g1'] = solv1[0]
        sample['g2'] = solv2[0]
        sample['g3'] = solv3[0]
        sample['intra_hb1'] = solv1[3]
        sample['intra_hb2'] = solv2[3]
        sample['intra_hb3'] = solv3[3]
        sample['inter_hb12'] = min(solv1[1],solv2[2]) + min(solv1[2],solv2[1])
        sample['inter_hb13'] = min(solv1[1],solv3[2]) + min(solv1[2],solv3[1])
        sample['inter_hb23'] = min(solv2[1],solv3[2]) + min(solv2[2],solv3[1])
        sample['solv1_x'] = data['solv1_x']
        sample['solv2_x'] = data['solv2_x']
        sample['gamma1'] = data['solv1_gamma']
        sample['gamma2'] = data['solv2_gamma']
        sample['gamma3'] = data['solv3_gamma']
        return sample

    def generate_sample(self,chemical_list,smiles_list,solv1_x,solv2_x,gamma_list=None):
        solvent_data = {}
        for i,sml in enumerate(smiles_list):
            solvent_data[chemical_list[i]] = []
            sml = Chem.MolToSmiles(Chem.MolFromSmiles(sml))
            mol = Chem.MolFromSmiles(sml)
            solvent_data[chemical_list[i]].append(mol_to_bigraph(mol,add_self_loop=True,
                                               node_featurizer=CanonicalAtomFeaturizer(),
                                               edge_featurizer=None,
                                               canonical_atom_order=False,
                                               explicit_hydrogens=False,
                                               num_virtual_nodes=0
                                               ))
            hba = Chem.rdMolDescriptors.CalcNumHBA(mol)
            hbd = Chem.rdMolDescriptors.CalcNumHBD(mol)
            solvent_data[chemical_list[i]].append(hba)
            solvent_data[chemical_list[i]].append(hbd)
            solvent_data[chemical_list[i]].append(min(hba,hbd))
        sample = {}
        solv1 = solvent_data[chemical_list[0]]
        solv2 = solvent_data[chemical_list[1]]
        solv3 = solvent_data[chemical_list[2]]
        sample['g1'] = solv1[0]
        sample['g2'] = solv2[0]
        sample['g3'] = solv3[0]
        sample['intra_hb1'] = solv1[3]
        sample['intra_hb2'] = solv2[3]
        sample['intra_hb3'] = solv3[3]
        sample['inter_hb12'] = min(solv1[1],solv2[2]) + min(solv1[2],solv2[1])
        sample['inter_hb13'] = min(solv1[1],solv3[2]) + min(solv1[2],solv3[1])
        sample['inter_hb23'] = min(solv2[1],solv3[2]) + min(solv2[2],solv3[1])
        sample['solv1_x'] = solv1_x
        sample['solv2_x'] = solv2_x
        if gamma_list:         
            sample['gamma1'] = gamma_list[0]
            sample['gamma2'] = gamma_list[1]
            sample['gamma3'] = gamma_list[2]
        return sample

    
    def generate_all(self):
        for solvent_id in self.solvent_smiles:
            mol = Chem.MolFromSmiles(self.solvent_smiles[solvent_id])
            self.solvent_data[solvent_id] = []
            self.solvent_data[solvent_id].append(mol_to_bigraph(mol,add_self_loop=True,
                                                                node_featurizer=CanonicalAtomFeaturizer(),
                                                                edge_featurizer=None,
                                                                canonical_atom_order=False,
                                                                explicit_hydrogens=False,
                                                                num_virtual_nodes=0
                                                                ))
            hba = Chem.rdMolDescriptors.CalcNumHBA(mol)
            hbd = Chem.rdMolDescriptors.CalcNumHBD(mol)
            self.solvent_data[solvent_id].append(hba)
            self.solvent_data[solvent_id].append(hbd)
            self.solvent_data[solvent_id].append(min(hba,hbd))
            
    def node_to_atom_list(self, idx):
        g1 = self.__getitem__(idx)['g1']
        g2 = self.__getitem__(idx)['g2']
        allowable_set = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca',
                         'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn',
                         'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au',
                         'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb']
        node_feat1 = g1.ndata['h'].numpy()[:,0:len(allowable_set)]
        atom_list1 = []
        for i in range(g1.number_of_nodes()):
            atom_list1.append(allowable_set[np.where(node_feat1[i]==1)[0][0]])
        node_feat2 = g2.ndata['h'].numpy()[:,0:len(allowable_set)]
        atom_list2 = []
        for i in range(g2.number_of_nodes()):
            atom_list2.append(allowable_set[np.where(node_feat2[i]==1)[0][0]])
        return {'atom_list': [atom_list1, atom_list2]}

    def get_smiles(self, idx):
        return {'smiles':[self.dataset['solv1_smiles'].iloc[idx],
                          self.dataset['solv2_smiles'].iloc[idx],
                          self.dataset['solv3_smiles'].iloc[idx]]}
    
    def generate_solvsys(self,batch_size=5):
        n_solv = 3
        solvsys = dgl.DGLGraph()
        solvsys.add_nodes(n_solv*batch_size)
        src = torch.arange(batch_size)
        dst = torch.arange(batch_size,2*batch_size)
        solvsys.add_edges(torch.cat((src,dst)),torch.cat((dst,src)))
        src = torch.arange(batch_size)
        dst = torch.arange(2*batch_size,3*batch_size)
        solvsys.add_edges(torch.cat((src,dst)),torch.cat((dst,src)))
        src = torch.arange(batch_size,2*batch_size)
        dst = torch.arange(2*batch_size,3*batch_size)
        solvsys.add_edges(torch.cat((src,dst)),torch.cat((dst,src)))
        solvsys.add_edges(torch.arange(n_solv*batch_size),torch.arange(n_solv*batch_size))    
        return solvsys


def collate_solvent_ternary(batch):
    keys = list(batch[0].keys())[3:]
    samples = list(map(lambda sample: sample.values(), batch))
    samples = list(map(list,zip(*samples)))
    batched_sample = {}
    batched_sample['g1'] = dgl.batch(samples[0])
    batched_sample['g2'] = dgl.batch(samples[1])
    batched_sample['g3'] = dgl.batch(samples[2])
    for i,key in enumerate(keys):        
        batched_sample[key] = torch.tensor(samples[i+3])
        batched_sample[key] = torch.tensor(samples[i+3])
    return batched_sample
   
def collate_solvent_binary(batch):
    keys = list(batch[0].keys())[2:]
    samples = list(map(lambda sample: sample.values(), batch))
    samples = list(map(list,zip(*samples)))
    batched_sample = {}
    batched_sample['g1'] = dgl.batch(samples[0])
    batched_sample['g2'] = dgl.batch(samples[1])
    for i,key in enumerate(keys):        
        batched_sample[key] = torch.tensor(samples[i+2])
        batched_sample[key] = torch.tensor(samples[i+2])
    return batched_sample
