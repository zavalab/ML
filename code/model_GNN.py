# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 22:28:22 2021

@author: sqin34
"""

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import dgl
from dgl.nn.pytorch import GraphConv, NNConv
import torch.nn.functional as F

def get_n_params(model):
    n_params = 0
    for item in list(model.parameters()):
        item_param = 1
        for dim in list(item.size()):
            item_param = item_param*dim
        n_params += item_param
    return n_params

class MPNNconv(nn.Module):
    def __init__(self, node_in_feats, edge_in_feats, node_out_feats=128,
                 edge_hidden_feats=32, num_step_message_passing=6):
        super(MPNNconv, self).__init__()

        self.project_node_feats = nn.Sequential(
            nn.Linear(node_in_feats, node_out_feats),
            nn.ReLU()
        )
        self.num_step_message_passing = num_step_message_passing
        edge_network = nn.Sequential(
            nn.Linear(edge_in_feats, edge_hidden_feats),
            nn.ReLU(),
            nn.Linear(edge_hidden_feats, node_out_feats * node_out_feats)
        )
        self.gnn_layer = NNConv(
            in_feats=node_out_feats,
            out_feats=node_out_feats,
            edge_func=edge_network,
            aggregator_type='sum'
        )
        self.gru = nn.GRU(node_out_feats, node_out_feats)

    def reset_parameters(self):
        self.project_node_feats[0].reset_parameters()
        self.gnn_layer.reset_parameters()
        for layer in self.gnn_layer.edge_func:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()
        self.gru.reset_parameters()

    def forward(self, g, node_feats, edge_feats):
        node_feats = self.project_node_feats(node_feats) # (V, node_out_feats)
        hidden_feats = node_feats.unsqueeze(0)           # (1, V, node_out_feats)

        for _ in range(self.num_step_message_passing):
            node_feats = F.relu(self.gnn_layer(g, node_feats, edge_feats))
            node_feats, hidden_feats = self.gru(node_feats.unsqueeze(0), hidden_feats)
            node_feats = node_feats.squeeze(0)
        return node_feats
    
class solvgnn_binary(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(solvgnn_binary, self).__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.global_conv1 = MPNNconv(node_in_feats=hidden_dim,
                                     edge_in_feats=1,
                                     node_out_feats=hidden_dim,
                                     edge_hidden_feats=32,
                                     num_step_message_passing=1)
        self.classify1 = nn.Linear(hidden_dim*2, hidden_dim)
        self.classify2 = nn.Linear(hidden_dim, hidden_dim)
        self.classify3 = nn.Linear(hidden_dim, n_classes)

    def forward(self, solvdata, empty_solvsys):
        g1 = solvdata['g1']
        g2 = solvdata['g2']
        with g1.local_scope():
            with g2.local_scope():
                
                h1 = g1.ndata['h'].float().cuda()
                h2 = g2.ndata['h'].float().cuda()
                solv1x = solvdata['solv1_x'].float().cuda()
                solv2x = 1 - solv1x
                inter_hb = solvdata['inter_hb'].float().cuda()
                intra_hb1 = solvdata['intra_hb1'].float().cuda()
                intra_hb2 = solvdata['intra_hb2'].float().cuda()
                
                h1_temp = F.relu(self.conv1(g1, h1))
                h1_temp = F.relu(self.conv2(g1, h1_temp))
                h2_temp = F.relu(self.conv1(g2, h2))
                h2_temp = F.relu(self.conv2(g2, h2_temp))
                g1.ndata['h'] = h1_temp
                g2.ndata['h'] = h2_temp
                
                hg1 = dgl.mean_nodes(g1, 'h')
                hg2 = dgl.mean_nodes(g2, 'h')
                hg1 = solv1x[:,None]*hg1
                hg2 = solv2x[:,None]*hg2
        
                hg = self.global_conv1(empty_solvsys, 
                                       torch.cat((hg1,hg2),axis=0), 
                                       torch.cat((inter_hb.repeat(2),intra_hb1,intra_hb2)).unsqueeze(1))
                hg = torch.cat((hg[0:len(hg)//2,:],hg[len(hg)//2:,:]),axis=1)
                output = F.relu(self.classify1(hg))
                output = F.relu(self.classify2(output))
                output = self.classify3(output)                        
                        
            return output    

class solvgnn_ternary(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(solvgnn_ternary, self).__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.global_conv1 = MPNNconv(node_in_feats=hidden_dim,
                                     edge_in_feats=1,
                                     node_out_feats=hidden_dim,
                                     edge_hidden_feats=32,
                                     num_step_message_passing=1)
        self.classify1 = nn.Linear(hidden_dim*3, hidden_dim)
        self.classify2 = nn.Linear(hidden_dim, hidden_dim)
        self.classify3 = nn.Linear(hidden_dim, n_classes)

    def forward(self, solvdata, empty_solvsys):

        g1 = solvdata['g1']
        g2 = solvdata['g2']
        g3 = solvdata['g3']
        with g1.local_scope():
            with g2.local_scope():
                with g3.local_scope():
                    h1 = g1.ndata['h'].float().cuda()
                    h2 = g2.ndata['h'].float().cuda()
                    h3 = g3.ndata['h'].float().cuda()
                    solv1x = solvdata['solv1_x'].float().cuda()
                    solv2x = solvdata['solv2_x'].float().cuda()
                    solv3x = 1 - solv1x - solv2x
                    inter_hb12 = solvdata['inter_hb12'].float().cuda()
                    inter_hb13 = solvdata['inter_hb13'].float().cuda()
                    inter_hb23 = solvdata['inter_hb23'].float().cuda()
                    intra_hb1 = solvdata['intra_hb1'].float().cuda()
                    intra_hb2 = solvdata['intra_hb2'].float().cuda()
                    intra_hb3 = solvdata['intra_hb3'].float().cuda()
                    
                    h1_temp = F.relu(self.conv1(g1, h1))
                    h1_temp = F.relu(self.conv2(g1, h1_temp))
                    h2_temp = F.relu(self.conv1(g2, h2))
                    h2_temp = F.relu(self.conv2(g2, h2_temp))
                    h3_temp = F.relu(self.conv1(g3, h3))
                    h3_temp = F.relu(self.conv2(g3, h3_temp))
                    g1.ndata['h'] = h1_temp
                    g2.ndata['h'] = h2_temp
                    g3.ndata['h'] = h3_temp        
            
                    hg1 = dgl.mean_nodes(g1, 'h')
                    hg2 = dgl.mean_nodes(g2, 'h')
                    hg3 = dgl.mean_nodes(g3, 'h')
                    hg1 = solv1x[:,None]*hg1
                    hg2 = solv2x[:,None]*hg2
                    hg3 = solv3x[:,None]*hg3
            
                    hg = self.global_conv1(empty_solvsys, 
                                           torch.cat((hg1,hg2,hg3),axis=0), 
                                           torch.cat((inter_hb12.repeat(2),inter_hb13.repeat(2),
                                                      inter_hb23.repeat(2),
                                                      intra_hb1,intra_hb2,intra_hb3)).unsqueeze(1))
                    hg = torch.cat((hg[0:len(hg)//3,:],hg[len(hg)//3:2*len(hg)//3,:],hg[2*len(hg)//3:,:]),axis=1)
                    output = F.relu(self.classify1(hg))
                    output = F.relu(self.classify2(output))
                    output = self.classify3(output)
            
                    return output