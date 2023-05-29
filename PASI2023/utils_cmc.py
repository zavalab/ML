import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import GraphConv
from rdkit import Chem
from molecular_graph import mol_to_bigraph
from atom_feat_encoding import CanonicalAtomFeaturizer 


def smiles_encoder(smiles, max_len, smi2index, unique_char):
    smiles_matrix = np.zeros((len(unique_char), max_len))
    for index, char in enumerate(smiles):
        smiles_matrix[smi2index[char], index] = 1
    return smiles_matrix


def generate_one_hot_encoding(df):
    periodic_elements = [
        "Ac", "Al", "Am", "Sb", "Ar", "As", "At", "Ba", "Bk", "Be", "Bi", "Bh", "B", "Br", "Cd", "Ca", "Cf", "C",
        "Ce", "Cs", "Cl", "Cr", "Co", "Cn", "Cu", "Cm", "Ds", "Db", "Dy", "Es", "Er", "Eu", "Fm", "Fl", "F", "Fr",
        "Gd", "Ga", "Ge", "Au", "Hf", "Hs", "He", "Ho", "H", "In", "I", "Ir", "Fe", "Kr", "La", "Lr", "Pb", "Li",
        "Lv", "Lu", "Mg", "Mn", "Mt", "Md", "Hg", "Mo", "Mc", "Nd", "Ne", "Np", "Ni", "Nh", "Nb", "N", "No", "Og",
        "Os", "O", "Pd", "P", "Pt", "Pu", "Po", "K", "Pr", "Pm", "Pa", "Ra", "Rn", "Re", "Rh", "Rg", "Rb", "Ru",
        "Rf", "Sm", "Sc", "Sg", "Se", "Si", "Ag", "Na", "Sr", "S", "Ta", "Tc", "Te", "Ts", "Tb", "Tl", "Th",
        "Tm", "Sn", "Ti", "W", "U", "V", "Xe", "Yb", "Y", "Zn", "Zr"]
    unique_chars = set(df.SMILES.apply(list).sum())
    upper_chars = []
    lower_chars = []
    for entry in unique_chars:
        if entry.isalpha():
            if entry.isupper():
                upper_chars.append(entry)
            elif entry.islower():
                lower_chars.append(entry)
    two_char_elements = []
    for upper in upper_chars:
        for lower in lower_chars:
            ch = upper + lower
            if ch in periodic_elements:
                two_char_elements.append(ch)
    two_char_elements_smiles = set()
    for char in two_char_elements:
        if df.SMILES.str.contains(char).any():
            two_char_elements_smiles.add(char)
    replace_dict = {"Cl": "L", "Br": "R", "Li": "X", "Na": "Z"}
    if df.SMILES.str.contains("Sc").any():
        print(
            'Warning: "Sc" element is found in the data set, since the element is rarely found '
            "in the drugs so we are not converting  "
            'it to single letter element, instead considering "S" '
            'and "c" as separate elements. '
        )
    df["processed_smiles"] = df.SMILES.copy()
    for pattern, repl in replace_dict.items():
        df["processed_smiles"] = df["processed_smiles"].str.replace(
            pattern, repl
        )
    unique_char = set(df.SMILES.apply(list).sum())
    longest_smiles = max(df.SMILES, key=len)
    smiles_maxlen = len(longest_smiles)
    ohe_all = []
    smi2index = {char: index for index, char in enumerate(unique_char)}
    for sml in df.SMILES.to_list():
        ohe_all.append(smiles_encoder(sml, smiles_maxlen, smi2index, unique_char))
    ohe = np.array(ohe_all)
    return ohe, smi2index, smiles_maxlen


class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        if  torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')
        self.conv = GraphConv(in_feats, h_feats)
        if torch.cuda.is_available():
            self.conv = self.conv.cuda()
        self.readout1 = nn.Linear(h_feats, h_feats).to(self.device)
        self.readout2 = nn.Linear(h_feats, num_classes).to(self.device)

    def forward(self, g, in_feat):
        with g.local_scope():
            h = self.conv(g, in_feat)
            h = F.relu(h)
            g.ndata['h'] = h
            meanNodes = dgl.mean_nodes(g, 'h')
            output = F.relu(self.readout1(meanNodes))
            output = self.readout2(output)
        return output


class cmc_dataset:  
    def __init__(self, input_file_path = "./Data/cmc_metadata.csv"):
        self.metadata = pd.read_csv(input_file_path)
        if  torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')
        self.graphs = {}
        self.generate()

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.graphs[idx]

    def generate(self):
        for idx, smiles in enumerate(self.metadata.SMILES.to_list()):
            mol = Chem.MolFromSmiles(smiles)
            self.graphs[idx] = []
            self.graphs[idx].append(mol_to_bigraph(mol,add_self_loop=True,
                                    node_featurizer=CanonicalAtomFeaturizer(),
                                    edge_featurizer=None,
                                    canonical_atom_order=False,
                                    explicit_hydrogens=False,
                                    num_virtual_nodes=0
                                    ).to(self.device))
            self.graphs[idx].append(torch.tensor(self.metadata.logCMC[idx]).to(self.device))


def collate(samples):
    if  torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph.to(device), torch.tensor(labels).unsqueeze(-1).to(device)