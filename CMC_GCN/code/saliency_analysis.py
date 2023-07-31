# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 13:54:36 2021

@author: sqin34
"""


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import pandas as pd
import numpy as np
from generate_graph_dataset import graph_dataset
from my_mol_visualizer import SmilesVisualizer
from chainer_chemistry.saliency.visualizer.visualizer_utils import abs_max_scaler
from model_GNN import GCNReg



dat = pd.read_csv("../data/dataset_202.csv", header=None)
sml_exp = dat[0].to_list()
logCMC = dat[1].to_numpy()

model_path = "../saved_models/"
model_name = "gnn_logs_save_202_hu256_lr0.005_best_trainalles_seed4592"
checkpoint = torch.load("../{}/{}".format(model_path, model_name))
model = GCNReg(74,256,1,True).cuda()
model.load_state_dict(checkpoint['model_state_dict'])

g_exp = graph_dataset(sml_exp,logCMC)

for test_id in range(len(sml_exp)):
    
    sml = sml_exp[test_id]
    cmc = logCMC[test_id]
    test_g = g_exp[test_id][0]
    n_feat = test_g.ndata['h'].numpy()
    pred,grad = model(test_g)
    pred = pred.cpu().detach().numpy().flatten()[0]
    n_sal = grad.cpu().detach().numpy()
    n_sal_sum = np.sum(n_sal*n_feat,axis=1)
    n_sal_sum_atom = np.sum(n_sal[:,0:43]*n_feat[:,0:43],axis=1)

    
    visualizer = SmilesVisualizer()
    scaler = abs_max_scaler
    svg,scaler_coeff = visualizer.visualize(n_sal_sum_atom, sml, save_filepath="../all_saliency/idx_nolab{}.png".format(str(test_id).zfill(3)), 
                                       visualize_ratio=1, bond_color=False,
                                       scaler=scaler)                                       
#                                       scaler=scaler, legend=sml+", true:{:.2f}, pred:{:.2f}".format(cmc,pred))
#    display(SVG(svg.replace('svg:', '')))
