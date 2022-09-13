# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 22:26:12 2021

@author: sqin34
"""

import os
import pandas as pd
import torch
import dgl
from solvgnn.util.atom_feat_encoding import CanonicalAtomFeaturizer 
from solvgnn.util.molecular_graph import mol_to_bigraph
from rdkit import Chem,DataStructs
from rdkit.Chem.Draw import MolToFile
from rdkit.Chem import AllChem
import numpy as np
from solvgnn.util.antoine_coeff import get_antoine_coef
import matplotlib
import matplotlib.pyplot as plt
import ternary
import win32com.client as win32
import time
    
def biPxy(x1, solv1_gam, solv2_gam, solv1_psat, solv2_psat):
    solv1_p = x1 * np.exp(solv1_gam)*solv1_psat
    solv2_p = (1-x1) * np.exp(solv2_gam)*solv2_psat
    equi_p = solv1_p+solv2_p
    y1 = solv1_p / equi_p
    return y1, equi_p

def data_aug_binary(df):
    df_flipped = df.copy()
    df_flipped["solv1"] = df["solv2"]
    df_flipped["solv2"] = df["solv1"]
    df_flipped["solv1_x"] = df["solv2_x"]
    df_flipped["solv2_x"] = df["solv1_x"]
    df_flipped["solv1_gamma"] = df["solv2_gamma"]
    df_flipped["solv2_gamma"] = df["solv1_gamma"]
    df_flipped["solv1_smiles"] = df["solv2_smiles"]
    df_flipped["solv2_smiles"] = df["solv1_smiles"]
    df_flipped["solv1_name"] = df["solv2_name"]
    df_flipped["solv2_name"] = df["solv1_name"]
    flip_rows = np.random.choice([True,False], size=len(df))
    df_new = pd.concat((df[~flip_rows],df_flipped[flip_rows])).sort_index()
    return df_new

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
    def predict(self,idx,model,empty_solvsys=None):
        sample = self.__getitem__(idx).copy()
        sample['solv1_x'] = torch.tensor(sample['solv1_x']).unsqueeze(0)
        sample['intra_hb1'] = torch.tensor(sample['intra_hb1']).unsqueeze(0)
        sample['intra_hb2'] = torch.tensor(sample['intra_hb2']).unsqueeze(0)
        sample['inter_hb'] = torch.tensor(sample['inter_hb']).unsqueeze(0)
        true = np.array([sample['gamma1'],sample['gamma2']])
        if empty_solvsys:
            predict = model(sample,empty_solvsys).detach().cpu().numpy().squeeze()
        else:
            predict = model(sample).detach().cpu().numpy().squeeze()
        return true,predict
    def predict_with_saliency(self,idx,model):
        empty_solvsys = self.generate_solvsys(batch_size=1)
        sample = self.__getitem__(idx).copy()
        sample['solv1_x'] = torch.tensor(sample['solv1_x']).unsqueeze(0)
        sample['intra_hb1'] = torch.tensor(sample['intra_hb1']).unsqueeze(0)
        sample['intra_hb2'] = torch.tensor(sample['intra_hb2']).unsqueeze(0)
        sample['inter_hb'] = torch.tensor(sample['inter_hb']).unsqueeze(0)
        true = np.array([sample['gamma1'],sample['gamma2']])        
        predict,saliency = model(sample,empty_solvsys)
        return true,predict.detach().cpu().numpy().squeeze(),saliency
    def predict_new(self,model,chemical_list,smiles_list,solv1_x,gamma_list=None):
        empty_solvsys = self.generate_solvsys(batch_size=1)
        sample = self.generate_sample(chemical_list,smiles_list,solv1_x,gamma_list)
        sample['solv1_x'] = torch.tensor(sample['solv1_x']).unsqueeze(0)
        sample['intra_hb1'] = torch.tensor(sample['intra_hb1']).unsqueeze(0)
        sample['intra_hb2'] = torch.tensor(sample['intra_hb2']).unsqueeze(0)
        sample['inter_hb'] = torch.tensor(sample['inter_hb']).unsqueeze(0)
        predict = model(sample,empty_solvsys).detach().cpu().numpy().squeeze()
        return predict
    def predict_new_with_uncertainty(self,model,cp_list,chemical_list,smiles_list,solv1_x,gamma_list=None):
        pred_all = []
        for checkpoint in cp_list:
            model.load_state_dict(torch.load(checkpoint)['model_state_dict'])
            pred_all.append(self.predict_new(model,chemical_list,smiles_list,solv1_x,gamma_list))
        pred_all = np.array(pred_all)
        return np.mean(pred_all,axis=0),np.std(pred_all,axis=0)    
    def get_VLE(self,model,chemical_list,smiles_list,solv1_x_range,temperature=298.,cp_list=None,uncertainty=False):
        solv1_gam = []
        solv2_gam = []
        solv1_ant = get_antoine_coef(chemical_list[0],temperature)
        solv2_ant = get_antoine_coef(chemical_list[1],temperature)
        if uncertainty:
            solv1_gam_std = []
            solv2_gam_std = []
            if cp_list:               
                for solv1_x in solv1_x_range:
                    pred,std = self.predict_new_with_uncertainty(model,cp_list,chemical_list,smiles_list,solv1_x,gamma_list=None)
                    solv1_gam.append(pred[0])
                    solv2_gam.append(pred[1])
                    solv1_gam_std.append(std[0])
                    solv2_gam_std.append(std[1])
                plot_data = pd.DataFrame({"solv1_gam":np.array(solv1_gam),
                         "solv1_gam_std":np.array(solv1_gam_std),
                         "solv2_gam":np.array(solv2_gam),
                         "solv2_gam_std":np.array(solv2_gam_std)})
            else:
                print("No Checkpoint List!")
        else:
            for solv1_x in solv1_x_range:
                pred = self.predict_new(model,chemical_list,smiles_list,solv1_x,gamma_list=None)
                solv1_gam.append(pred[0])
                solv2_gam.append(pred[1])
            plot_data = pd.DataFrame({"solv1_gam":np.array(solv1_gam),
                                      "solv2_gam":np.array(solv2_gam)})
        solv1_psat = 10 ** (solv1_ant[0] - solv1_ant[1]/(temperature + solv1_ant[2])) # return vapor pressure in bar
        solv2_psat = 10 ** (solv2_ant[0] - solv2_ant[1]/(temperature + solv2_ant[2]))
        y1,equi_p = biPxy(solv1_x_range,solv1_gam,solv2_gam,solv1_psat,solv2_psat)
        plot_data["x1"] = np.array(solv1_x_range)
        plot_data["y1"] = np.array(y1)
        plot_data["pressure"] = np.array(equi_p)
        return plot_data
    def get_VLE_from_aspen(self,chemical_list,npoint=50,temperature=298.):

        aspen = win32.Dispatch('Apwn.Document')
        aspen.InitFromArchive2(os.path.abspath('C:/Users/sqin34/OneDrive - UW-Madison/Research/Miscibility/aspen/test_pxy.bkp'))
        
        aspen.Tree.FindNode('/Data/Components/Specifications/Input/DBNAME1/SOLV1').Value = chemical_list[0].upper()
        aspen.Tree.FindNode('/Data/Components/Specifications/Input/DBNAME1/SOLV2').Value = chemical_list[1].upper()
        
        aspen.Tree.FindNode('/Data/Properties/Analysis/BINRY-1/Input/CNPOINT').Value = npoint
        aspen.Tree.FindNode('/Data/Properties/Analysis/BINRY-1/Input/TLIST/#0').Value = temperature
        
        # aspen.Reinit()
        aspen.Engine.Run2()
        temperature = []
        x1 = []
        y1 = []
        solv1_gam = []
        solv2_gam = []
        equi_p = []

        for i in range(npoint+1):
            x1.append(aspen.Tree.FindNode('/Data/Properties/Analysis/BINRY-1/Output/Prop Data/PROPTAB/LIQUID1 MOLEFRAC SOLV1/{}'.format(i+1)).Value)
            y1.append(aspen.Tree.FindNode('/Data/Properties/Analysis/BINRY-1/Output/Prop Data/PROPTAB/VAPOR MOLEFRAC SOLV1/{}'.format(i+1)).Value)
            solv1_gam.append(aspen.Tree.FindNode('/Data/Properties/Analysis/BINRY-1/Output/Prop Data/PROPTAB/LIQUID1 GAMMA SOLV1/{}'.format(i+1)).Value)
            solv2_gam.append(aspen.Tree.FindNode('/Data/Properties/Analysis/BINRY-1/Output/Prop Data/PROPTAB/LIQUID1 GAMMA SOLV2/{}'.format(i+1)).Value)
            equi_p.append(aspen.Tree.FindNode('/Data/Properties/Analysis/BINRY-1/Output/Prop Data/PROPTAB/TOTAL PRES/{}'.format(i+1)).Value)
        aspen.Close()
        
        return pd.DataFrame({"aspen_x1":np.array(x1),
                             "aspen_y1":np.array(y1),
                             "aspen_solv1_gam":np.log(np.array(solv1_gam)),
                             "aspen_solv2_gam":np.log(np.array(solv2_gam)),
                             "aspen_pressure":np.array(equi_p)})
        
    def plot_VLE(self,model,chemical_list,smiles_list,solv1_x_range,temperature=298.,
                 cp_list=None,uncertainty=False,overlay_with_aspen=False,save_dir=None,save_figure=True,
                 save_data=False,aspen_chemical_list=None,convert_cosmo=False,plot_data_cosmo=None,
                 overlay_with_cosmo=False,overlay_with_aspen_and_cosmo=False):
        if convert_cosmo:
            if plot_data_cosmo is not None:
                plot_data_cosmo = plot_data_cosmo.rename(columns={" Liquid Phase  Mole Fraction x1":"cosmo_x1",
                                                " Gas Phase  Mole Fraction y1":"cosmo_y1",
                                                " Total  Pressure ":"cosmo_pressure",
                                                " Activity Coeff.  ln(?1) ":"cosmo_solv1_gam",
                                                " Activity Coeff.  ln(?2) ":"cosmo_solv2_gam"})
                plot_data_cosmo["cosmo_pressure"] = plot_data_cosmo["cosmo_pressure"]/1000
                plot_data_cosmo = plot_data_cosmo[["cosmo_x1","cosmo_y1","cosmo_solv1_gam","cosmo_solv2_gam",
                                                   "cosmo_pressure"]]
            else:
                print("Please specify cosmo data to convert!")
        start_time = time.time()
        print('solv1: {}, solv2: {}'.format(chemical_list[0].lower(), chemical_list[1].lower()))
        print('temperature: {} K'.format(temperature))
        print('number of data points: {}'.format(len(solv1_x_range)))
        print('---start GNN prediction---')
        st_gnn = time.time()
        plot_data = self.get_VLE(model,chemical_list,smiles_list,solv1_x_range,temperature=temperature,
                                 cp_list=cp_list,uncertainty=uncertainty)
        print('---finished in {:.2f} seconds---'.format(time.time()-st_gnn))
            
        fig,ax = plt.subplots(figsize=(3,2.8))
        ax.grid(color='lightgray',linewidth=0.75,alpha=0.5)
        if overlay_with_aspen:
            if aspen_chemical_list == None:
                aspen_chemical_list = chemical_list.copy()                
            print('---start aspen Plus simulation---')
            st_aspen = time.time()
            plot_data_aspen = self.get_VLE_from_aspen(aspen_chemical_list,npoint=len(solv1_x_range)-1,temperature=temperature)
            print('---finished in {:.2f} seconds---'.format(time.time()-st_aspen))
            print('---start plotting Pxy---')
            plot_data = pd.concat([plot_data,plot_data_aspen],axis=1)
            ax.plot(plot_data["aspen_x1"],plot_data["aspen_pressure"],label="Aspen",color='black',
                    marker='s',markersize=4,markerfacecolor="none")
            ax.plot(plot_data["aspen_y1"],plot_data["aspen_pressure"],label="Aspen",color='black',
                    marker='o',markersize=4,markerfacecolor="none")            
            ax.scatter(plot_data["x1"],plot_data["pressure"],color='black',marker='s',label="SolvGNN",s=16)
            ax.scatter(plot_data["y1"],plot_data["pressure"],color='black',marker='o',label="SolvGNN",s=16)
        elif overlay_with_cosmo:
            plot_data = pd.concat([plot_data,plot_data_cosmo],axis=1)
            ax.plot(plot_data["cosmo_x1"],plot_data["cosmo_pressure"],label="COSMO-RS",color='black',
                    marker='s',markersize=4,markerfacecolor="none")
            ax.plot(plot_data["cosmo_y1"],plot_data["cosmo_pressure"],label="COSMO-RS",color='black',
                    marker='o',markersize=4,markerfacecolor="none")            
            ax.scatter(plot_data["x1"],plot_data["pressure"],color='black',label="SolvGNN",marker='s',s=16)
            ax.scatter(plot_data["y1"],plot_data["pressure"],color='black',label="SolvGNN",marker='o',s=16)      
        elif overlay_with_aspen_and_cosmo:
            if aspen_chemical_list == None:
                aspen_chemical_list = chemical_list.copy()                
            print('---start aspen Plus simulation---')
            st_aspen = time.time()
            plot_data_aspen = self.get_VLE_from_aspen(aspen_chemical_list,npoint=len(solv1_x_range)-1,temperature=temperature)
            print('---finished in {:.2f} seconds---'.format(time.time()-st_aspen))
            print('---start plotting Pxy---')
            plot_data = pd.concat([plot_data,plot_data_aspen,plot_data_cosmo],axis=1)
            ax.plot(plot_data["aspen_x1"],plot_data["aspen_pressure"],label="Aspen",color='black')
            ax.plot(plot_data["aspen_y1"],plot_data["aspen_pressure"],label="Aspen",color='black')            
            ax.scatter(plot_data["cosmo_x1"],plot_data["cosmo_pressure"],label="COSMO-RS",
                    marker='s',s=16,facecolors="none",edgecolors='black')
            ax.scatter(plot_data["cosmo_y1"],plot_data["cosmo_pressure"],label="COSMO-RS",
                    marker='o',s=16,facecolors="none",edgecolors='black')                   
            ax.scatter(plot_data["x1"],plot_data["pressure"],color='black',label="SolvGNN",marker='s',s=16)
            ax.scatter(plot_data["y1"],plot_data["pressure"],color='black',label="SolvGNN",marker='o',s=16)      
        else:
            ax.plot(plot_data["x1"],plot_data["pressure"],label="$liquid$",color='black',marker='s',markersize=4)
            ax.plot(plot_data["y1"],plot_data["pressure"],label="$vapor$",color='black',marker='o',markersize=4)
        ax.set_xticks(np.arange(0,1.2,0.2))
        ax.set_xlabel("$x_1$, $y_1$")
        ax.set_ylabel("Pressure (bar)")
        ax.set_title("$P_{xy}$"+" for {}(1) and {}(2)".format(chemical_list[0],chemical_list[1]))
        ax.legend(loc='best')
        plt.tight_layout()
        if save_figure:
            print('---start saving figure---')
            plt.savefig(save_dir + 'pxy_{}_{}.svg'.format(chemical_list[0].lower(),
                                                           chemical_list[1].lower()),
                        pad_inches=0,dpi=400,transparent=True)
        plt.show()
        if save_data:
            print('---start saving data---')
            plot_data.to_csv(save_dir + "pxy_{}_{}.csv".format(chemical_list[0].lower(),
                                                                chemical_list[1].lower()),
                                                                index=False)
        print('---overall finished in {:.2f} seconds---'.format(time.time()-start_time))
        return plot_data
    def plot_gamma(self,chemical_list,plot_data,save_dir=None,save_figure=True,overlay_with_aspen=True,
                   overlay_with_cosmo=True,overlay_with_aspen_and_cosmo=True):
        fig,ax = plt.subplots(figsize=(3,2.8))
        ax.grid(color='lightgray',linewidth=0.75,alpha=0.5)
        if overlay_with_aspen_and_cosmo:
            ax.plot(plot_data["aspen_x1"],plot_data["aspen_solv1_gam"],c="black",label="UNIFAC $\gamma_1$")
            ax.scatter(plot_data["cosmo_x1"],plot_data["cosmo_solv1_gam"],label="COSMO-RS",marker="s",facecolors="none",edgecolors="black",s=16)
            ax.errorbar(plot_data["x1"].iloc[0],plot_data["solv1_gam"].iloc[0],yerr=plot_data["solv1_gam_std"].iloc[0],color="black",fmt="x",markersize=4,label="SolvGNN")
            ax.errorbar(plot_data["x1"].iloc[1:],plot_data["solv1_gam"].iloc[1:],yerr=plot_data["solv1_gam_std"].iloc[1:],color="black",fmt="s",markersize=5)
            ax.plot(plot_data["aspen_x1"],plot_data["aspen_solv2_gam"],c="r",label="UNIFAC $\gamma_2$")
            ax.scatter(plot_data["cosmo_x1"],plot_data["cosmo_solv2_gam"],label="COSMO-RS",marker="s",facecolors="none",edgecolors="r",s=16)
            ax.errorbar(plot_data["x1"].iloc[:-1],plot_data["solv2_gam"].iloc[:-1],yerr=plot_data["solv2_gam_std"].iloc[:-1],color="r",fmt="s",markersize=4,label="SolvGNN")
            ax.errorbar(plot_data["x1"].iloc[-1],plot_data["solv2_gam"].iloc[-1],yerr=plot_data["solv2_gam_std"].iloc[-1],color="r",fmt="x",markersize=5)
        elif overlay_with_aspen:
            ax.plot(plot_data["aspen_x1"],plot_data["aspen_solv1_gam"],c="black",label="UNIFAC $\gamma_1$")
            ax.errorbar(plot_data["x1"].iloc[0],plot_data["solv1_gam"].iloc[0],yerr=plot_data["solv1_gam_std"].iloc[0],color="black",fmt="x",markersize=4,label="SolvGNN")
            ax.errorbar(plot_data["x1"].iloc[1:],plot_data["solv1_gam"].iloc[1:],yerr=plot_data["solv1_gam_std"].iloc[1:],color="black",fmt="s",markersize=5)
            ax.plot(plot_data["aspen_x1"],plot_data["aspen_solv2_gam"],c="r",label="UNIFAC $\gamma_2$")
            ax.errorbar(plot_data["x1"].iloc[:-1],plot_data["solv2_gam"].iloc[:-1],yerr=plot_data["solv2_gam_std"].iloc[:-1],color="r",fmt="s",markersize=4,label="SolvGNN")
            ax.errorbar(plot_data["x1"].iloc[-1],plot_data["solv2_gam"].iloc[-1],yerr=plot_data["solv2_gam_std"].iloc[-1],color="r",fmt="x",markersize=5)            
        elif overlay_with_cosmo:
            ax.plot(plot_data["cosmo_x1"],plot_data["cosmo_solv1_gam"],label="COSMO-RS",marker="s",facecolors="none",edgecolors="black",s=16)
            ax.errorbar(plot_data["x1"].iloc[0],plot_data["solv1_gam"].iloc[0],yerr=plot_data["solv1_gam_std"].iloc[0],color="black",fmt="x",markersize=4,label="SolvGNN")
            ax.errorbar(plot_data["x1"].iloc[1:],plot_data["solv1_gam"].iloc[1:],yerr=plot_data["solv1_gam_std"].iloc[1:],color="black",fmt="s",markersize=5)
            ax.plot(plot_data["cosmo_x1"],plot_data["cosmo_solv2_gam"],label="COSMO-RS",marker="s",facecolors="none",edgecolors="r",s=16)
            ax.errorbar(plot_data["x1"].iloc[:-1],plot_data["solv2_gam"].iloc[:-1],yerr=plot_data["solv2_gam_std"].iloc[:-1],color="r",fmt="s",markersize=4,label="SolvGNN")
            ax.errorbar(plot_data["x1"].iloc[-1],plot_data["solv2_gam"].iloc[-1],yerr=plot_data["solv2_gam_std"].iloc[-1],color="r",fmt="x",markersize=5)
        else:
            ax.errorbar(plot_data["x1"].iloc[0],plot_data["solv1_gam"].iloc[0],yerr=plot_data["solv1_gam_std"].iloc[0],color="black",fmt="x",markersize=4,label="SolvGNN $\gamma_1$")
            ax.errorbar(plot_data["x1"].iloc[1:],plot_data["solv1_gam"].iloc[1:],yerr=plot_data["solv1_gam_std"].iloc[1:],color="black",fmt="s",markersize=5)
            ax.errorbar(plot_data["x1"].iloc[:-1],plot_data["solv2_gam"].iloc[:-1],yerr=plot_data["solv2_gam_std"].iloc[:-1],color="r",fmt="s",markersize=4,label="SolvGNN $\gamma_2$")
            ax.errorbar(plot_data["x1"].iloc[-1],plot_data["solv2_gam"].iloc[-1],yerr=plot_data["solv2_gam_std"].iloc[-1],color="r",fmt="x",markersize=5)            
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$\ln\gamma_i$")
        ax.legend()
        plt.tight_layout()
        if save_figure:
            print('---start saving figure---')
            plt.savefig(save_dir + 'pxy_{}_{}_gamma.svg'.format(chemical_list[0].lower(),
                                                                 chemical_list[1].lower()),
                        pad_inches=0,dpi=400,transparent=True)
        plt.show()
        return
    def get_fp(self,idx=None,smiles_list=None,radius=3,fpsize=1024,return_matrix=False):
        if idx:
            smiles_list = self.get_smiles(idx)['smiles']            
        fp_mat = []
        if return_matrix is False:
            for sml in smiles_list:
                mol = Chem.MolFromSmiles(sml)
                fp_mat.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=fpsize))
        else:
            for sml in smiles_list:
                mol = Chem.MolFromSmiles(sml)
                fp_mat.append(np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=fpsize)))
            fp_mat = np.array(fp_mat)
        return fp_mat
    def get_internal_similarity(self,idx=None,smiles_list=None,metric=DataStructs.TanimotoSimilarity):
        fp_mat = self.get_fp(idx=idx,smiles_list=smiles_list)
        return DataStructs.FingerprintSimilarity(fp_mat[0],fp_mat[1], metric=metric)
    def get_external_similarity(self,idx1=None,idx2=None,
                                smiles_list1=None,smiles_list2=None,
                                metric=DataStructs.TanimotoSimilarity):
        if idx1 and idx2:
            smiles_list1 = self.get_smiles(idx1)['smiles']
            smiles_list2 = self.get_smiles(idx2)['smiles']
        fp_mat1 = self.get_fp(idx=idx1,smiles_list=smiles_list1)
        fp_mat2 = self.get_fp(idx=idx2,smiles_list=smiles_list2)
        score = [(DataStructs.FingerprintSimilarity(fp_mat1[0],fp_mat2[0], metric=metric)+\
                  DataStructs.FingerprintSimilarity(fp_mat1[1],fp_mat2[1], metric=metric))/2,
                 (DataStructs.FingerprintSimilarity(fp_mat1[0],fp_mat2[1], metric=metric)+\
                  DataStructs.FingerprintSimilarity(fp_mat1[1],fp_mat2[0], metric=metric))/2]
        best_sim_perm = np.argmax(score)
        return best_sim_perm,score[best_sim_perm]
    def get_chemical_structure(self,idx,save_dir):
        smiles = self.get_smiles(idx)['smiles']
        for i,sml in enumerate(smiles):
            mol = Chem.MolFromSmiles(sml)
            MolToFile(mol,save_dir + "id{}_{}.svg".format(idx,i+1))
        return
    def get_all_predictions(self,base_solv1_id,model,cp_list,
                            solv1_x_range=[0.1,0.3,0.5,0.7,0.9],
                            save_dir=None,save_data=True):
        base_solv1_smiles = self.solvent_smiles[base_solv1_id]
        base_solv1_name = self.solvent_names[base_solv1_id]
        solv2_smiles = []
        solv2_name = []
        solv1_x_all = []
        pred_gam1 = []
        pred_gam2 = []
        count = 0
        for solv_id in self.solvent_smiles:
            for solv1_x in solv1_x_range:
                solv_name = self.solvent_names[solv_id]
                solv_smiles = self.solvent_smiles[solv_id]
                solv2_name.append(solv_name)
                solv2_smiles.append(solv_smiles)
                solv1_x_all.append(solv1_x)
                pred,_ = self.predict_new_with_uncertainty(model,cp_list,
                                                              [base_solv1_name,solv_name],
                                                              [base_solv1_smiles,solv_smiles],
                                                              solv1_x,gamma_list=None)
                pred_gam1.append(pred[0])
                pred_gam2.append(pred[1])
                count += 1
                if count % 100 == 0:
                    print('{} out of {} done!'.format(count,len(self.solvent_smiles)*len(solv1_x_range)))
        output_cv = pd.DataFrame({"solv1_smiles":base_solv1_smiles,
                                  "solv2_smiles":solv2_smiles,
                                  "solv1_name":base_solv1_name,
                                  "solv2_name":solv2_name,
                                  "solv1":base_solv1_id,
                                  "solv2":np.repeat(list(self.solvent_smiles),len(solv1_x_range)),
                                  "solv1_x":solv1_x_all,
                                  "pred_gam1":pred_gam1,
                                  "pred_gam2":pred_gam2})
        if save_data:
            output_cv.to_csv(save_dir + "output_cv_base_{}.csv".format(base_solv1_id),index=False)
        return output_cv



class solvent_dataset_binary_edge1:

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
        sample['intra_hb1'] = 1
        sample['intra_hb2'] = 1
        sample['inter_hb'] = 1
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
    def predict(self,idx,model,empty_solvsys=None):
        sample = self.__getitem__(idx).copy()
        sample['solv1_x'] = torch.tensor(sample['solv1_x']).unsqueeze(0)
        sample['intra_hb1'] = torch.tensor(sample['intra_hb1']).unsqueeze(0)
        sample['intra_hb2'] = torch.tensor(sample['intra_hb2']).unsqueeze(0)
        sample['inter_hb'] = torch.tensor(sample['inter_hb']).unsqueeze(0)
        true = np.array([sample['gamma1'],sample['gamma2']])
        if empty_solvsys:
            predict = model(sample,empty_solvsys).detach().cpu().numpy().squeeze()
        else:
            predict = model(sample).detach().cpu().numpy().squeeze()
        return true,predict
    def predict_new(self,model,chemical_list,smiles_list,solv1_x,gamma_list=None):
        empty_solvsys = self.generate_solvsys(batch_size=1)
        sample = self.generate_sample(chemical_list,smiles_list,solv1_x,gamma_list)
        sample['solv1_x'] = torch.tensor(sample['solv1_x']).unsqueeze(0)
        sample['intra_hb1'] = torch.tensor(sample['intra_hb1']).unsqueeze(0)
        sample['intra_hb2'] = torch.tensor(sample['intra_hb2']).unsqueeze(0)
        sample['inter_hb'] = torch.tensor(sample['inter_hb']).unsqueeze(0)
        predict = model(sample,empty_solvsys).detach().cpu().numpy().squeeze()
        return predict


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
    def predict(self,idx,model,empty_solvsys=None):
        sample = self.__getitem__(idx).copy()
        sample['intra_hb1'] = torch.tensor(sample['intra_hb1']).unsqueeze(0)
        sample['intra_hb2'] = torch.tensor(sample['intra_hb2']).unsqueeze(0)
        sample['intra_hb3'] = torch.tensor(sample['intra_hb3']).unsqueeze(0)
        sample['inter_hb12'] = torch.tensor(sample['inter_hb12']).unsqueeze(0)
        sample['inter_hb13'] = torch.tensor(sample['inter_hb13']).unsqueeze(0)
        sample['inter_hb23'] = torch.tensor(sample['inter_hb23']).unsqueeze(0)
        sample['solv1_x'] = torch.tensor(sample['solv1_x']).unsqueeze(0)
        sample['solv2_x'] = torch.tensor(sample['solv2_x']).unsqueeze(0)
        true = np.array([sample['gamma1'],sample['gamma2'],sample['gamma3']])
        if empty_solvsys:
            predict = model(sample,empty_solvsys).detach().cpu().numpy().squeeze()
        else:
            predict = model(sample).detach().cpu().numpy().squeeze()
        return true,predict
    def predict_new(self,model,chemical_list,smiles_list,solv1_x,solv2_x,gamma_list=None):
        empty_solvsys = self.generate_solvsys(batch_size=1)
        sample = self.generate_sample(chemical_list,smiles_list,solv1_x,solv2_x,gamma_list)
        sample['solv1_x'] = torch.tensor(sample['solv1_x']).unsqueeze(0)
        sample['solv2_x'] = torch.tensor(sample['solv2_x']).unsqueeze(0)
        sample['intra_hb1'] = torch.tensor(sample['intra_hb1']).unsqueeze(0)
        sample['intra_hb2'] = torch.tensor(sample['intra_hb2']).unsqueeze(0)
        sample['intra_hb3'] = torch.tensor(sample['intra_hb3']).unsqueeze(0)
        sample['inter_hb12'] = torch.tensor(sample['inter_hb12']).unsqueeze(0)
        sample['inter_hb13'] = torch.tensor(sample['inter_hb13']).unsqueeze(0)
        sample['inter_hb23'] = torch.tensor(sample['inter_hb23']).unsqueeze(0)
        predict = model(sample,empty_solvsys).detach().cpu().numpy().squeeze()
        return predict
    def predict_new_with_uncertainty(self,model,cp_list,chemical_list,smiles_list,solv1_x,solv2_x,gamma_list=None):
        pred_all = []
        for checkpoint in cp_list:
            model.load_state_dict(torch.load(checkpoint)['model_state_dict'])
            pred_all.append(self.predict_new(model,chemical_list,smiles_list,solv1_x,solv2_x,gamma_list))
        pred_all = np.array(pred_all)
        return np.mean(pred_all,axis=0),np.std(pred_all,axis=0)
    def get_VLE(self,model,chemical_list,smiles_list,solv1_x_range,solv2_x_range,
                temperature=298.,cp_list=None,uncertainty=False):
        solv1_gam = []
        solv2_gam = []
        solv3_gam = []
        solv1_ant = get_antoine_coef(chemical_list[0],temperature)
        solv2_ant = get_antoine_coef(chemical_list[1],temperature)
        solv3_ant = get_antoine_coef(chemical_list[2],temperature)
        solv1_psat = 10 ** (solv1_ant[0] - solv1_ant[1]/(temperature + solv1_ant[2])) # return vapor pressure in bar
        solv2_psat = 10 ** (solv2_ant[0] - solv2_ant[1]/(temperature + solv2_ant[2]))
        solv3_psat = 10 ** (solv3_ant[0] - solv3_ant[1]/(temperature + solv3_ant[2]))
        x_list = sample_ternary_x(solv1_x_range,solv2_x_range)
        print('number of data points: {}'.format(len(x_list)))
        if uncertainty:
            solv1_gam_std = []
            solv2_gam_std = []
            solv3_gam_std = []
            if cp_list:
                for (solv1_x,solv2_x,_) in x_list:
                    pred,std = self.predict_new_with_uncertainty(model,cp_list,chemical_list,smiles_list,solv1_x,solv2_x,gamma_list=None)
                    solv1_gam.append(pred[0])
                    solv2_gam.append(pred[1])
                    solv3_gam.append(pred[2])
                    solv1_gam_std.append(std[0])
                    solv2_gam_std.append(std[1])
                    solv3_gam_std.append(std[2])
                plot_data = pd.DataFrame({"solv1_gam":np.array(solv1_gam),
                                         "solv1_gam_std":np.array(solv1_gam_std),
                                         "solv2_gam":np.array(solv2_gam),
                                         "solv2_gam_std":np.array(solv2_gam_std),
                                         "solv3_gam":np.array(solv3_gam),
                                         "solv3_gam_std":np.array(solv3_gam_std)})
            else:
                print("No Checkpoint List!")
        else:
            for (solv1_x,solv2_x,_) in x_list:
                pred = self.predict_new(model,chemical_list,smiles_list,solv1_x,solv2_x,gamma_list=None)
                solv1_gam.append(pred[0])
                solv2_gam.append(pred[1])
                solv3_gam.append(pred[2])
            plot_data = pd.DataFrame({"solv1_gam":np.array(solv1_gam),
                         "solv2_gam":np.array(solv2_gam),
                         "solv3_gam":np.array(solv3_gam)})
        y_list,equi_p = terPxy(x_list[:,0],x_list[:,1],solv1_gam,solv2_gam,solv3_gam,solv1_psat,solv2_psat,solv3_psat)
        plot_data["x1"] = x_list[:,0]
        plot_data["x2"] = x_list[:,1]
        plot_data["x3"] = x_list[:,2]
        plot_data["y1"] = y_list[0,:]
        plot_data["y2"] = y_list[1,:]
        plot_data["y3"] = y_list[2,:]
        plot_data["pressure"] = np.array(equi_p)
        return plot_data
    def get_VLE_from_ASPEN(self,chemical_list,selected_pressure,ntieline=5,temperature=298,
                           plot_aspen=False):

        aspen = win32.Dispatch('Apwn.Document')
        aspen.InitFromArchive2(os.path.abspath('C:/Users/sqin34/OneDrive - UW-Madison/Research/Miscibility/aspen_ternary/test_ternary.bkp'))
        
        aspen.Tree.FindNode('/Data/Components/Specifications/Input/DBNAME1/SOLV1').Value = chemical_list[0].upper()
        aspen.Tree.FindNode('/Data/Components/Specifications/Input/DBNAME1/SOLV2').Value = chemical_list[1].upper()
        aspen.Tree.FindNode('/Data/Components/Specifications/Input/DBNAME1/SOLV2').Value = chemical_list[2].upper()
        
        aspen.Tree.FindNode('/Data/Properties/Analysis/TERDI-1/Input/NTIELINE').Value = ntieline
        aspen.Tree.FindNode('/Data/Properties/Analysis/TERDI-1/Input/PRES').Value = selected_pressure
        aspen.Tree.FindNode('/Data/Properties/Analysis/TERDI-1/Input/TEMP').Value = temperature
        
        # aspen.Reinit()
        aspen.Engine.Run2()
        tielines = []
        azeotrope = {}
        
        for i in range(ntieline):
            tielines.append([(
                aspen.Tree.FindNode('/Data/Properties/Analysis/TERDI-1/Output/Prop Data/TIEN/MOLEFRAC LIQUID1 SOLV1/{}'.format(i+1)).Value,
                aspen.Tree.FindNode('/Data/Properties/Analysis/TERDI-1/Output/Prop Data/TIEN/MOLEFRAC LIQUID1 SOLV2/{}'.format(i+1)).Value,
                aspen.Tree.FindNode('/Data/Properties/Analysis/TERDI-1/Output/Prop Data/TIEN/MOLEFRAC LIQUID1 SOLV3/{}'.format(i+1)).Value),
                (aspen.Tree.FindNode('/Data/Properties/Analysis/TERDI-1/Output/Prop Data/TIEN/MOLEFRAC LIQUID2 SOLV1/{}'.format(i+1)).Value,
                aspen.Tree.FindNode('/Data/Properties/Analysis/TERDI-1/Output/Prop Data/TIEN/MOLEFRAC LIQUID2 SOLV2/{}'.format(i+1)).Value,
                aspen.Tree.FindNode('/Data/Properties/Analysis/TERDI-1/Output/Prop Data/TIEN/MOLEFRAC LIQUID2 SOLV3/{}'.format(i+1)).Value)])                
        try:
            aze_temp = aspen.Tree.FindNode('/Data/Properties/Analysis/TERDI-1/Output/Prop Data/AZET/TEMP/1').Value
            azeotrope[0] = [aze_temp,
                            aspen.Tree.FindNode('/Data/Properties/Analysis/TERDI-1/Output/Prop Data/AZET/MOLEFRAC SOLV1/1').Value,
                            aspen.Tree.FindNode('/Data/Properties/Analysis/TERDI-1/Output/Prop Data/AZET/MOLEFRAC SOLV2/1').Value,
                            aspen.Tree.FindNode('/Data/Properties/Analysis/TERDI-1/Output/Prop Data/AZET/MOLEFRAC SOLV3/1').Value]
            print("Azeotrope found in ASPEN!")
        except:
            print("No azeotrope found in ASPEN!")
        aspen.Close()
        
        if plot_aspen==True:
            print('---start plotting phase diagram from Aspen---')
            scale = 1.0
            figure, tax = ternary.figure(scale=scale)
            figure.set_size_inches(4, 4)
            tax.boundary(linewidth=1.0)
            tax.gridlines(color="black", multiple=1)
            tax.gridlines(color="gray", multiple=0.2, linewidth=0.5)
            offset = 0.14
            tax.left_axis_label('{}'.format(chemical_list[2].lower()), offset=offset)
            tax.right_axis_label('{}'.format(chemical_list[1].lower()), offset=offset)
            tax.bottom_axis_label('{}'.format(chemical_list[0].lower()))
            for tieline in tielines: 
                tax.line(tieline[0],tieline[1],markersize=4,
                         color="black",marker="s",markerfacecolor="none")
            tax.ticks(axis='lbr', linewidth=1, multiple=0.2, tick_formats="%.1f", offset=0.02)
            tax.set_background_color(color="white", alpha=0) # the detault, essentially
            tax.clear_matplotlib_ticks()
            tax.get_axes().axis('off')
            tax.get_axes().set_aspect(1)
            tax._redraw_labels()
            ternary.plt.tight_layout()     
        return {"tielines":tielines,"azeotrope":azeotrope}
        
    def plot_VLE(self,model,chemical_list,smiles_list,
                 solv1_x_range=np.arange(0,1.25,0.25),
                 solv2_x_range=np.arange(0,1.25,0.25),
                 temperature=298.,
                 cp_list=None,uncertainty=False,overlay_with_Aspen=False,
                 save_dir=None,save_figure=True,
                 save_data=True,aspen_chemical_list=None):
        start_time = time.time()
        print('solv1: {}, solv2: {}, solv3: {}'.format(chemical_list[0].lower(),
                                                       chemical_list[1].lower(),
                                                       chemical_list[2].lower()))
        print('temperature: {} K'.format(temperature))
        print('---start GNN prediction---')
        st_gnn = time.time()
        plot_data = self.get_VLE(model,chemical_list,smiles_list,solv1_x_range,solv2_x_range,
                                 temperature=temperature,cp_list=cp_list,uncertainty=uncertainty)
        print('---finished in {:.2f} seconds---'.format(time.time()-st_gnn))       
        print('---start plotting Pxy---')
        scale = 1.0
        figure, tax = ternary.figure(scale=scale)
        figure.set_size_inches(4, 4)
        tax.boundary(linewidth=1.0)
        tax.gridlines(color="black", multiple=1)
        tax.gridlines(color="gray", multiple=0.2, linewidth=0.5)
        offset = 0.14
        tax.left_axis_label('{}'.format(chemical_list[2].lower()), offset=offset)
        tax.right_axis_label('{}'.format(chemical_list[1].lower()), offset=offset)
        tax.bottom_axis_label('{}'.format(chemical_list[0].lower()))
        tax.scatter([tuple(x) for x in plot_data[["x1","x2","x3"]].to_numpy()],s=20,
                    c=plot_data["pressure"],cmap='Greys',marker="h",
                    vmin=plot_data["pressure"].min(),vmax=plot_data["pressure"].max(),
                    colorbar=True,colormap="Greys",cbarlabel="Pressure (Bar)",cb_kwargs={"shrink":0.7})
        tax.ticks(axis='lbr', linewidth=1, multiple=0.2, tick_formats="%.1f", offset=0.02)
        tax.set_background_color(color="white", alpha=0) # the detault, essentially
        tax.clear_matplotlib_ticks()
        tax.get_axes().axis('off')
        tax.get_axes().set_aspect(1)
        tax._redraw_labels()
        ternary.plt.tight_layout()
        if save_figure:
            print('---start saving figure---')
            plt.savefig(save_dir = 'pxy_{}_{}_{}.svg'.format(chemical_list[0].lower(),
                                                           chemical_list[1].lower(),
                                                           chemical_list[2].lower()),
                        pad_inches=0,dpi=400,transparent=True)
        ternary.plt.show()
        if save_data:
            print('---start saving data---')
            plot_data.to_csv(save_dir + "pxy_{}_{}_{}.csv".format(chemical_list[0].lower(),
                                                                chemical_list[1].lower(),
                                                                chemical_list[2].lower()),
                                                                index=False)
        print('---overall finished in {:.2f} seconds---'.format(time.time()-start_time))
        return plot_data
    def replot_VLE(self,chemical_list,plot_data,save_dir=None,save_figure=False,
                   selected_pressure_list=None,selected_pressure_buffer=0.01,
                   convert_cosmo=False,marker_size=20,specific_range=False,
                   background_pressure=False,
                   overlay_with_cosmo=False,plot_data_cosmo=None):      
        if convert_cosmo:
            marker_size=100
            if plot_data_cosmo is not None:
                plot_data_cosmo["x1"] = plot_data_cosmo[" Liquid Phase  Mole Fraction x1"]
                plot_data_cosmo["x2"] = plot_data_cosmo[" Liquid Phase  Mole Fraction x2"]
                plot_data_cosmo["x3"] = plot_data_cosmo[" Liquid Phase  Mole Fraction x3"]
                plot_data_cosmo["pressure"] = plot_data_cosmo[" Total  Pressure "]/1000      
            else:
                plot_data["x1"] = plot_data[" Liquid Phase  Mole Fraction x1"]
                plot_data["x2"] = plot_data[" Liquid Phase  Mole Fraction x2"]
                plot_data["x3"] = plot_data[" Liquid Phase  Mole Fraction x3"]
                plot_data["pressure"] = plot_data[" Total  Pressure "]/1000
        if specific_range == False:
            specific_range = [plot_data["pressure"].min(),
                              plot_data["pressure"].max()]
        print('---start plotting Pxy---')
        scale = 1.0
        figure, tax = ternary.figure(scale=scale)
        figure.set_size_inches(4, 4)
        tax.boundary(linewidth=1.0)
        tax.gridlines(color="black", multiple=1)
        tax.gridlines(color="gray", multiple=0.2, linewidth=0.5)
        offset = 0.14
        tax.left_axis_label('{}'.format(chemical_list[2].lower()), offset=offset)
        tax.right_axis_label('{}'.format(chemical_list[1].lower()), offset=offset)
        tax.bottom_axis_label('{}'.format(chemical_list[0].lower()))
        if background_pressure:          
            tax.scatter([tuple(x) for x in plot_data[["x1","x2","x3"]].to_numpy()],s=marker_size,
                        c=plot_data["pressure"],cmap='Greys',marker="h",
                        vmin=specific_range[0],vmax=specific_range[1],
                        colorbar=True,colormap="Greys",cbarlabel="Pressure (Bar)",cb_kwargs={"shrink":0.7})
        if selected_pressure_list:
            color_list = matplotlib.cm.get_cmap('tab10')
            if overlay_with_cosmo==True:
                if plot_data_cosmo is not None:
                    for i,selected_pressure in enumerate(selected_pressure_list):
                        subset_cosmo = plot_data_cosmo[(plot_data_cosmo["pressure"]<selected_pressure*(1+selected_pressure_buffer))\
                                           &(plot_data_cosmo["pressure"]>selected_pressure*(1-selected_pressure_buffer))]
                        tax.scatter([tuple(x) for x in subset_cosmo[["x1","x2","x3"]].to_numpy()],s=16,
                                    label="P$\sim${:.2f} bar".format(selected_pressure),marker="s",edgecolors=color_list(i),facecolors='none')
                else:
                    print("Please specify cosmo data to overlay with!")
            for i,selected_pressure in enumerate(selected_pressure_list):
                subset = plot_data[(plot_data["pressure"]<selected_pressure*(1+selected_pressure_buffer))\
                                   &(plot_data["pressure"]>selected_pressure*(1-selected_pressure_buffer))]
                tax.scatter([tuple(x) for x in subset[["x1","x2","x3"]].to_numpy()],s=16,
                           label="P$\sim${:.2f} bar".format(selected_pressure),marker="s",color=color_list(i))
            tax.legend(loc="upper left",fontsize="small")
        tax.ticks(axis='lbr', linewidth=1, multiple=0.2, tick_formats="%.1f", offset=0.02)
        tax.set_background_color(color="white", alpha=0) # the detault, essentially
        tax.clear_matplotlib_ticks()
        tax.get_axes().axis('off')
        tax.get_axes().set_aspect(1)
        tax._redraw_labels()
        ternary.plt.tight_layout()
        if save_figure:
            print('---start saving figure---')
            plt.savefig(save_dir + 'pxy_{}_{}_{}_replot.svg'.format(chemical_list[0].lower(),
                                                           chemical_list[1].lower(),
                                                           chemical_list[2].lower()),
                        pad_inches=0,dpi=400,transparent=True)
        ternary.plt.show()
        return figure,tax


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
   
def collate_solvent(batch):
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
