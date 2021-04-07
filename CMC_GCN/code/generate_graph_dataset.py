# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 01:49:12 2020

@author: sqin34
"""
import torch
import dgl
from dgllife.utils import BaseAtomFeaturizer,CanonicalAtomFeaturizer,CanonicalBondFeaturizer 
from dgllife.utils import mol_to_graph,mol_to_bigraph,mol_to_complete_graph,smiles_to_complete_graph
from rdkit import Chem
from rdkit.Chem import Draw
import numpy as np

__all__ = ['graph_dataset']

class graph_dataset(object):

    def __init__(self, smiles, y, 
                 node_enc = CanonicalAtomFeaturizer(), edge_enc = None,
                 graph_type = mol_to_bigraph, canonical_atom_order = False):
        super(graph_dataset, self).__init__()
#        self.num_graphs = num_graphs
        self.smiles = smiles
        self.y = y
        self.graph_type = graph_type
        self.node_enc = node_enc
        self.edge_enc = edge_enc
        self.canonical_atom_order = canonical_atom_order
        self.graphs = []
        self.labels = []
        self._generate()

    def __len__(self):
        """Return the number of graphs in the dataset."""
        return len(self.graphs)

    def __getitem__(self, idx):
        """Get the i^th sample.

        Paramters
        ---------
        idx : int
            The sample index.

        Returns
        -------
        (dgl.DGLGraph, int)
            The graph and its label.
        """
        return self.graphs[idx], self.labels[idx]

#    @property
#    def num_classes(self):
#        """Number of classes."""
#        return 8
    def getsmiles(self, idx):
        return self.smiles[idx]
    
    def node_to_atom(self, idx):
        g = self.graphs[idx]
        allowable_set = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca',
                         'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn',
                         'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au',
                         'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb']
        node_feat = g.ndata['h'].numpy()[:,0:len(allowable_set)]
        atom_list = []
        for i in range(g.number_of_nodes()):
            atom_list.append(allowable_set[np.where(node_feat[i]==1)[0][0]])
        return atom_list
    
    def _generate(self):
        if self.graph_type==mol_to_bigraph:
            for i,j in enumerate(self.smiles):
                m = Chem.MolFromSmiles(j)
    #            m = Chem.AddHs(m)
                g = self.graph_type(m,True,self.node_enc,self.edge_enc,
                                    self.canonical_atom_order)
                self.graphs.append(g)
                self.labels.append(torch.tensor(self.y[i]))
        elif self.graph_type==smiles_to_complete_graph:
            for i,j in enumerate(self.smiles):
                g = self.graph_type(j,True,self.node_enc,self.edge_enc,
                                    self.canonical_atom_order)
                self.graphs.append(g)
                self.labels.append(torch.tensor(self.y[i]))
                

def summarize_graph_data(g):
    node_data = g.ndata['h'].numpy()
    print("node data:\n",node_data)
    edge_data = g.edata
    print("edge data:",edge_data)
    adj_mat = g.adjacency_matrix_scipy(transpose=True,return_edge_ids=False)
    adj_mat = adj_mat.todense().astype(np.float32)
    print("adjacency matrix:\n",adj_mat)


def collate(samples):
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels).unsqueeze(-1)