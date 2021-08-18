# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 14:27:52 2020

@author: sqin34
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 11:54:36 2020

@author: sqin34
"""

# https://docs.dgl.ai/en/0.4.x/tutorials/basics/4_batch.html

import dgl
import torch
from dgl.nn.pytorch import GraphConv

import dgl.function as fn
import torch.nn as nn
import torch.nn.functional as F

def collate(samples):
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels)

###############################################################################

class GCNReg(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, saliency=False):
        super(GCNReg, self).__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
#        self.conv3 = GraphConv(hidden_dim, hidden_dim)
        self.classify1 = nn.Linear(hidden_dim, hidden_dim)
        self.classify2 = nn.Linear(hidden_dim, hidden_dim)
        self.classify3 = nn.Linear(hidden_dim, n_classes)
        self.saliency = saliency
#        self.classify3 = nn.Linear(hidden_dim, n_classes)
# default 2 conv 3 classify
    def forward(self, g):
        # Use node degree as the initial node feature. For undirected graphs, the in-degree
        # is the same as the out_degree.
#        h_deg = g.in_degrees().view(-1,1).float()
#        h = g.ndata['h'].view(-1,1).float().cuda()
        h = g.ndata['h'].float().cuda()
#        h_feat = g.ndata['h'].float()
#        h = torch.cat((h_deg, h_feat), axis=-1).cuda()
#        g.ndata['h'] = torch.cat((h_deg, h_feat), axis=-1)
        # Perform graph convolution and activation function.
        if self.saliency == True:
            h.requires_grad = True
        h1 = F.relu(self.conv1(g, h))
        h1 = F.relu(self.conv2(g, h1))
#        h = F.relu(self.conv3(g, h))
#        h = F.relu(self.conv4(g, h))
#        h = F.relu(self.conv3(g, h))
        g.ndata['h'] = h1
        # Calculate graph representation by averaging all the node representations.
        hg = dgl.mean_nodes(g, 'h')
        output = F.relu(self.classify1(hg))
        output = F.relu(self.classify2(output))
#        output = F.relu(self.classify2(output))
        output = self.classify3(output)
        if self.saliency == True:
            output.backward()
            return output, h.grad
        else:
            return output

class GCNReg_1mlp(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, saliency=False):
        super(GCNReg_1mlp, self).__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
#        self.conv3 = GraphConv(hidden_dim, hidden_dim)
        self.classify1 = nn.Linear(hidden_dim, hidden_dim)
#        self.classify2 = nn.Linear(hidden_dim, hidden_dim)
        self.classify3 = nn.Linear(hidden_dim, n_classes)
        self.saliency = saliency
#        self.classify3 = nn.Linear(hidden_dim, n_classes)
# default 2 conv 3 classify
    def forward(self, g):
        # Use node degree as the initial node feature. For undirected graphs, the in-degree
        # is the same as the out_degree.
#        h_deg = g.in_degrees().view(-1,1).float()
#        h = g.ndata['h'].view(-1,1).float().cuda()
        h = g.ndata['h'].float().cuda()
#        h_feat = g.ndata['h'].float()
#        h = torch.cat((h_deg, h_feat), axis=-1).cuda()
#        g.ndata['h'] = torch.cat((h_deg, h_feat), axis=-1)
        # Perform graph convolution and activation function.
        if self.saliency == True:
            h.requires_grad = True
        h1 = F.relu(self.conv1(g, h))
        h1 = F.relu(self.conv2(g, h1))
#        h = F.relu(self.conv3(g, h))
#        h = F.relu(self.conv4(g, h))
#        h = F.relu(self.conv3(g, h))
        g.ndata['h'] = h1
        # Calculate graph representation by averaging all the node representations.
        hg = dgl.mean_nodes(g, 'h')
        output = F.relu(self.classify1(hg))
#        output = F.relu(self.classify2(output))
#        output = F.relu(self.classify2(output))
        output = self.classify3(output)
        if self.saliency == True:
            output.backward()
            return output, h.grad
        else:
            return output

class GCNReg_0mlp(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, saliency=False):
        super(GCNReg_0mlp, self).__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
#        self.conv3 = GraphConv(hidden_dim, hidden_dim)
#        self.classify1 = nn.Linear(hidden_dim, hidden_dim)
#        self.classify2 = nn.Linear(hidden_dim, hidden_dim)
        self.classify3 = nn.Linear(hidden_dim, n_classes)
        self.saliency = saliency
#        self.classify3 = nn.Linear(hidden_dim, n_classes)
# default 2 conv 3 classify
    def forward(self, g):
        # Use node degree as the initial node feature. For undirected graphs, the in-degree
        # is the same as the out_degree.
#        h_deg = g.in_degrees().view(-1,1).float()
#        h = g.ndata['h'].view(-1,1).float().cuda()
        h = g.ndata['h'].float().cuda()
#        h_feat = g.ndata['h'].float()
#        h = torch.cat((h_deg, h_feat), axis=-1).cuda()
#        g.ndata['h'] = torch.cat((h_deg, h_feat), axis=-1)
        # Perform graph convolution and activation function.
        if self.saliency == True:
            h.requires_grad = True
        h1 = F.relu(self.conv1(g, h))
        h1 = F.relu(self.conv2(g, h1))
#        h = F.relu(self.conv3(g, h))
#        h = F.relu(self.conv4(g, h))
#        h = F.relu(self.conv3(g, h))
        g.ndata['h'] = h1
        # Calculate graph representation by averaging all the node representations.
        hg = dgl.mean_nodes(g, 'h')
#        output = F.relu(self.classify1(hg))
#        output = F.relu(self.classify2(output))
#        output = F.relu(self.classify2(output))
        output = self.classify3(hg)
        if self.saliency == True:
            output.backward()
            return output, h.grad
        else:
            return output

class GCNReg_3mlp(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, saliency=False):
        super(GCNReg_3mlp, self).__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
#        self.conv3 = GraphConv(hidden_dim, hidden_dim)
        self.classify1 = nn.Linear(hidden_dim, hidden_dim)
        self.classify2 = nn.Linear(hidden_dim, hidden_dim)
        self.classify3 = nn.Linear(hidden_dim, hidden_dim)        
        self.classify4 = nn.Linear(hidden_dim, n_classes)
        self.saliency = saliency
#        self.classify3 = nn.Linear(hidden_dim, n_classes)
# default 2 conv 3 classify
    def forward(self, g):
        # Use node degree as the initial node feature. For undirected graphs, the in-degree
        # is the same as the out_degree.
#        h_deg = g.in_degrees().view(-1,1).float()
#        h = g.ndata['h'].view(-1,1).float().cuda()
        h = g.ndata['h'].float().cuda()
#        h_feat = g.ndata['h'].float()
#        h = torch.cat((h_deg, h_feat), axis=-1).cuda()
#        g.ndata['h'] = torch.cat((h_deg, h_feat), axis=-1)
        # Perform graph convolution and activation function.
        if self.saliency == True:
            h.requires_grad = True
        h1 = F.relu(self.conv1(g, h))
        h1 = F.relu(self.conv2(g, h1))
#        h = F.relu(self.conv3(g, h))
#        h = F.relu(self.conv4(g, h))
#        h = F.relu(self.conv3(g, h))
        g.ndata['h'] = h1
        # Calculate graph representation by averaging all the node representations.
        hg = dgl.mean_nodes(g, 'h')
        output = F.relu(self.classify1(hg))
        output = F.relu(self.classify2(output))
        output = F.relu(self.classify3(output))
        output = self.classify4(output)
        if self.saliency == True:
            output.backward()
            return output, h.grad
        else:
            return output

class GCNReg_1gc(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, saliency=False):
        super(GCNReg_1gc, self).__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim)
#        self.conv2 = GraphConv(hidden_dim, hidden_dim)
#        self.conv3 = GraphConv(hidden_dim, hidden_dim)
        self.classify1 = nn.Linear(hidden_dim, hidden_dim)
#        self.classify2 = nn.Linear(hidden_dim, hidden_dim)
        self.classify3 = nn.Linear(hidden_dim, n_classes)
        self.saliency = saliency
#        self.classify3 = nn.Linear(hidden_dim, n_classes)
# default 2 conv 3 classify
    def forward(self, g):
        # Use node degree as the initial node feature. For undirected graphs, the in-degree
        # is the same as the out_degree.
#        h_deg = g.in_degrees().view(-1,1).float()
#        h = g.ndata['h'].view(-1,1).float().cuda()
        h = g.ndata['h'].float().cuda()
#        h_feat = g.ndata['h'].float()
#        h = torch.cat((h_deg, h_feat), axis=-1).cuda()
#        g.ndata['h'] = torch.cat((h_deg, h_feat), axis=-1)
        # Perform graph convolution and activation function.
        if self.saliency == True:
            h.requires_grad = True
        h1 = F.relu(self.conv1(g, h))
#        h1 = F.relu(self.conv2(g, h1))
#        h = F.relu(self.conv3(g, h))
#        h = F.relu(self.conv4(g, h))
#        h = F.relu(self.conv3(g, h))
        g.ndata['h'] = h1
        # Calculate graph representation by averaging all the node representations.
        hg = dgl.mean_nodes(g, 'h')
        output = F.relu(self.classify1(hg))
#        output = F.relu(self.classify2(output))
#        output = F.relu(self.classify2(output))
        output = self.classify3(output)
        if self.saliency == True:
            output.backward()
            return output, h.grad
        else:
            return output

class GCNReg_3gc(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, saliency=False):
        super(GCNReg_3gc, self).__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.conv3 = GraphConv(hidden_dim, hidden_dim)
        self.classify1 = nn.Linear(hidden_dim, hidden_dim)
#        self.classify2 = nn.Linear(hidden_dim, hidden_dim)
        self.classify3 = nn.Linear(hidden_dim, n_classes)
        self.saliency = saliency
#        self.classify3 = nn.Linear(hidden_dim, n_classes)
# default 2 conv 3 classify
    def forward(self, g):
        # Use node degree as the initial node feature. For undirected graphs, the in-degree
        # is the same as the out_degree.
#        h_deg = g.in_degrees().view(-1,1).float()
#        h = g.ndata['h'].view(-1,1).float().cuda()
        h = g.ndata['h'].float().cuda()
#        h_feat = g.ndata['h'].float()
#        h = torch.cat((h_deg, h_feat), axis=-1).cuda()
#        g.ndata['h'] = torch.cat((h_deg, h_feat), axis=-1)
        # Perform graph convolution and activation function.
        if self.saliency == True:
            h.requires_grad = True
        h1 = F.relu(self.conv1(g, h))
        h1 = F.relu(self.conv2(g, h1))
        h1 = F.relu(self.conv3(g, h1))
#        h = F.relu(self.conv4(g, h))
#        h = F.relu(self.conv3(g, h))
        g.ndata['h'] = h1
        # Calculate graph representation by averaging all the node representations.
        hg = dgl.mean_nodes(g, 'h')
        output = F.relu(self.classify1(hg))
#        output = F.relu(self.classify2(output))
#        output = F.relu(self.classify2(output))
        output = self.classify3(output)
        if self.saliency == True:
            output.backward()
            return output, h.grad
        else:
            return output


class GCNReg_3mlpdo(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, saliency=False):
        super(GCNReg_3mlpdo, self).__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
#        self.conv3 = GraphConv(hidden_dim, hidden_dim)
        self.classify1 = nn.Linear(hidden_dim, hidden_dim//2)
        self.classify2 = nn.Linear(hidden_dim//2, hidden_dim//4)
        self.classify3 = nn.Linear(hidden_dim//4, hidden_dim//8)        
        self.classify4 = nn.Linear(hidden_dim//8, n_classes)
        self.saliency = saliency
#        self.classify3 = nn.Linear(hidden_dim, n_classes)
# default 2 conv 3 classify
    def forward(self, g):
        # Use node degree as the initial node feature. For undirected graphs, the in-degree
        # is the same as the out_degree.
#        h_deg = g.in_degrees().view(-1,1).float()
#        h = g.ndata['h'].view(-1,1).float().cuda()
        h = g.ndata['h'].float().cuda()
#        h_feat = g.ndata['h'].float()
#        h = torch.cat((h_deg, h_feat), axis=-1).cuda()
#        g.ndata['h'] = torch.cat((h_deg, h_feat), axis=-1)
        # Perform graph convolution and activation function.
        if self.saliency == True:
            h.requires_grad = True
        h1 = F.relu(self.conv1(g, h))
        h1 = F.relu(self.conv2(g, h1))
#        h = F.relu(self.conv3(g, h))
#        h = F.relu(self.conv4(g, h))
#        h = F.relu(self.conv3(g, h))
        g.ndata['h'] = h1
        # Calculate graph representation by averaging all the node representations.
        hg = dgl.mean_nodes(g, 'h')
        output = F.relu(self.classify1(hg))
        output = F.relu(self.classify2(output))
        output = F.relu(self.classify3(output))
        output = self.classify4(output)
        if self.saliency == True:
            output.backward()
            return output, h.grad
        else:
            return output


class GCNReg_print(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, saliency=False):
        super(GCNReg_print, self).__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
#        self.conv3 = GraphConv(hidden_dim, hidden_dim)
        self.classify1 = nn.Linear(hidden_dim, hidden_dim)
        self.classify2 = nn.Linear(hidden_dim, hidden_dim)
        self.classify3 = nn.Linear(hidden_dim, n_classes)
        self.saliency = saliency
#        self.classify3 = nn.Linear(hidden_dim, n_classes)
# default 2 conv 3 classify
    def forward(self, g):
        # Use node degree as the initial node feature. For undirected graphs, the in-degree
        # is the same as the out_degree.
#        h_deg = g.in_degrees().view(-1,1).float()
#        h = g.ndata['h'].view(-1,1).float().cuda()
        h = g.ndata['h'].float().cuda()
        print(h)
#        h_feat = g.ndata['h'].float()
#        h = torch.cat((h_deg, h_feat), axis=-1).cuda()
#        g.ndata['h'] = torch.cat((h_deg, h_feat), axis=-1)
        # Perform graph convolution and activation function.
        if self.saliency == True:
            h.requires_grad = True
        h1 = F.relu(self.conv1(g, h))
        print(h1)
        h1 = F.relu(self.conv2(g, h1))
        print(h1)
#        h = F.relu(self.conv3(g, h))
#        h = F.relu(self.conv4(g, h))
#        h = F.relu(self.conv3(g, h))
        g.ndata['h'] = h1
        # Calculate graph representation by averaging all the node representations.
        hg = dgl.mean_nodes(g, 'h')
        print(hg)
        output = F.relu(self.classify1(hg))
        print(output)
        output = F.relu(self.classify2(output))
        print(output)
#        output = F.relu(self.classify2(output))
        output = self.classify3(output)
        print(output)
        if self.saliency == True:
            output.backward()
            return output, h.grad
        else:
            return output



#class GCNReg(nn.Module):
#    def __init__(self, in_dim, hidden_dim, n_classes):
#        super(GCNReg, self).__init__()
#        self.conv1 = GraphConv(in_dim, hidden_dim)
#        self.conv2 = GraphConv(hidden_dim, hidden_dim)
##        self.conv3 = GraphConv(hidden_dim, hidden_dim)
#        self.classify1 = nn.Linear(hidden_dim, hidden_dim)
#        self.classify2 = nn.Linear(hidden_dim, hidden_dim)
#        self.classify3 = nn.Linear(hidden_dim, n_classes)
##        self.classify3 = nn.Linear(hidden_dim, n_classes)
## default 2 conv 3 classify
#    def forward(self, g):
#        # Use node degree as the initial node feature. For undirected graphs, the in-degree
#        # is the same as the out_degree.
##        h_deg = g.in_degrees().view(-1,1).float()
##        h = g.ndata['h'].view(-1,1).float().cuda()
#        h = g.ndata['h'].float().cuda()
##        h_feat = g.ndata['h'].float()
##        h = torch.cat((h_deg, h_feat), axis=-1).cuda()
##        g.ndata['h'] = torch.cat((h_deg, h_feat), axis=-1)
#        # Perform graph convolution and activation function.
#        h.requires_grad = True
#        g.ndata['g'] = h
#        h = F.relu(self.conv1(g, h))
#        h = F.relu(self.conv2(g, h))
##        h = F.relu(self.conv3(g, h))
##        h = F.relu(self.conv4(g, h))
##        h = F.relu(self.conv3(g, h))
#        g.ndata['h'] = h
#        # Calculate graph representation by averaging all the node representations.
#        hg = dgl.mean_nodes(g, 'h')
#        output = F.relu(self.classify1(hg))
#        output = F.relu(self.classify2(output))
##        output = F.relu(self.classify2(output))
#        output = self.classify3(output)
#        return output

#
#            degs = graph.out_degrees().to(feat.device).float().clamp(min=1)
#            norm = th.pow(degs, -0.5)
#            shp = norm.shape + (1,) * (feat.dim() - 1)
#            norm = th.reshape(norm, shp)
#            feat = feat * norm

class Classifier_v2_backup(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(Classifier_v2_backup, self).__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.conv3 = GraphConv(hidden_dim, hidden_dim)
        self.conv4 = GraphConv(hidden_dim, hidden_dim)
#        self.conv3 = GraphConv(hidden_dim, hidden_dim)
        self.classify1 = nn.Linear(hidden_dim, hidden_dim)
        self.classify2 = nn.Linear(hidden_dim, n_classes)
#        self.classify3 = nn.Linear(hidden_dim, n_classes)
#        self.classify3 = nn.Linear(hidden_dim, n_classes)
# default 2 conv 3 classify
    def forward(self, g):
        # Use node degree as the initial node feature. For undirected graphs, the in-degree
        # is the same as the out_degree.
#        h_deg = g.in_degrees().view(-1,1).float()
#        h = g.ndata['h'].view(-1,1).float().cuda()
        h = g.ndata['h'].float().cuda()
#        h_feat = g.ndata['h'].float()
#        h = torch.cat((h_deg, h_feat), axis=-1).cuda()
#        g.ndata['h'] = torch.cat((h_deg, h_feat), axis=-1)
        # Perform graph convolution and activation function.
        h.requires_grad = True
        g.ndata['g'] = h
        h = F.relu(self.conv1(g, h))
        h = F.relu(self.conv2(g, h))
        h = F.relu(self.conv3(g, h))
        h = F.relu(self.conv4(g, h))
#        h = F.relu(self.conv3(g, h))
        g.ndata['h'] = h
        # Calculate graph representation by averaging all the node representations.
        hg = dgl.mean_nodes(g, 'h')
        output = F.relu(self.classify1(hg))
#        output = F.relu(self.classify2(output))
#        output = F.relu(self.classify2(output))
        output = self.classify2(output)
        return output

 
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn

# Sends a message of node feature h.
msg = fn.copy_src(src='h', out='m')


def reduce(nodes):
    """Take an average over all neighbor node features hu and use it to
    overwrite the original node feature."""
    accum = torch.mean(nodes.mailbox['m'], 1)
    return {'h': accum}


class NodeApplyModule(nn.Module):
    """Update the node feature hv with ReLU(Whv+b)."""

    def __init__(self, in_feats, out_feats, activation):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation

    def forward(self, node):
        h = self.linear(node.data['h'])
        h = self.activation(h)
        return {'h': h}


class GCN(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(GCN, self).__init__()
        self.apply_mod = NodeApplyModule(in_feats, out_feats, activation)

    def forward(self, g, feature):
        # Initialize the node features with h.
        g.ndata['h'] = feature
        g.update_all(msg, reduce)
        g.apply_nodes(func=self.apply_mod)
        return g.ndata.pop('h')


class Net(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super(Net, self).__init__()

        self.layers = nn.ModuleList([
            GCN(in_dim, hidden_dim, F.relu),
            GCN(hidden_dim, hidden_dim, F.relu),
            GCN(hidden_dim, hidden_dim, F.relu)])
        self.classify1 = nn.Linear(hidden_dim, hidden_dim, F.relu)
        self.classify2 = nn.Linear(hidden_dim, 1)

    def forward(self, g):
        # For undirected graphs, in_degree is the same as
        # out_degree.
        h_degree = g.in_degrees().view(-1, 1).float()
        h_feat = g.ndata['h'].float()
        if h_feat.is_cuda:
            h = torch.cat((h_degree.cuda(), h_feat), axis=-1)
        else:
            h = torch.cat((h_degree, h_feat), axis=-1).cuda()
        h.requires_grad = True
        g.ndata['g'] = h
        for conv in self.layers:
            h = conv(g, h)
        g.ndata['h'] = h
        hg = dgl.mean_nodes(g, 'h')
        hg = self.classify1(hg)
        hg = self.classify1(hg)
        hg = self.classify2(hg)
        return hg


