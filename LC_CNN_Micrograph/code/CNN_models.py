# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 15:58:41 2021

@author: sqin34
"""
from __future__ import absolute_import


import torch.nn as nn
import torch.nn.functional as F


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def model_summary(model):
    print("Modules | Parameters")
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        print('{} | {}'.format(name,param))
        total_params+=param
    print(f"Total Trainable Params: {total_params}")
    return total_params

class CNNNet_BN_multiclass_alllev1(nn.Module):
    def __init__(self, n_class=1, n_channel=3):
        super(CNNNet_BN_multiclass_alllev1, self).__init__()
        self.conv1 = nn.Conv2d(n_channel,16,7,padding=3)
        self.pool = nn.AvgPool2d(7,7)
        self.conv1_bn = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16,32,7,padding=3)
        self.conv2_bn = nn.BatchNorm2d(32)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(800,64)
        self.fc1_bn = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64,32)
        self.fc2_bn = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32,n_class)
    
    def forward(self,x):
        x = self.pool(self.conv1_bn(F.relu(self.conv1(x))))
        x = self.pool(self.conv2_bn(F.relu(self.conv2(x))))
        x = self.flatten(x)
        x = self.fc1_bn(F.relu(self.fc1(x)))
        x = self.fc2_bn(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


class CNNNet_BN_multiclass_alllev2(nn.Module):
    def __init__(self, n_class=1, n_channel=3):
        super(CNNNet_BN_multiclass_alllev2, self).__init__()
        self.conv1 = nn.Conv2d(n_channel,32,7,padding=3)
        self.pool = nn.AvgPool2d(3,3)
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32,64,7,padding=3)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64,128,7,padding=3)
        self.conv3_bn = nn.BatchNorm2d(128)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(10368,128)
        self.fc1_bn = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128,64)
        self.fc2_bn = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64,n_class)
    
    def forward(self,x):
        x = self.pool(self.conv1_bn(F.relu(self.conv1(x))))
        x = self.pool(self.conv2_bn(F.relu(self.conv2(x))))
        x = self.pool(self.conv3_bn(F.relu(self.conv3(x))))
        x = self.flatten(x)
        x = self.fc1_bn(F.relu(self.fc1(x)))
        x = self.fc2_bn(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


class rdp_1dCNN(nn.Module):
    def __init__(self, input_channel=1,n_class=1):
        super(rdp_1dCNN, self).__init__()
        self.conv1 = nn.Conv1d(input_channel,8,5,padding=2)
        self.conv2 = nn.Conv1d(8,16,5,padding=2)
        self.conv3 = nn.Conv1d(16,32,5,padding=2)
        self.pool = nn.AvgPool1d(3)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(192,64)
        self.fc2 = nn.Linear(64,32)
        self.fc3 = nn.Linear(32,n_class)
        self.conv1_bn = nn.BatchNorm1d(8)
        self.conv2_bn = nn.BatchNorm1d(16)
        self.conv3_bn = nn.BatchNorm1d(32)
        self.fc1_bn = nn.BatchNorm1d(64)
        self.fc2_bn = nn.BatchNorm1d(32)
    
    def forward(self,x):
        x = self.pool(self.conv1_bn(F.relu(self.conv1(x))))
        x = self.pool(self.conv2_bn(F.relu(self.conv2(x))))
        x = self.pool(self.conv3_bn(F.relu(self.conv3(x))))
        x = self.flatten(x)
        x = self.fc1_bn(F.relu(self.fc1(x)))
        x = self.fc2_bn(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


class ec_1dCNN(nn.Module):
    def __init__(self, input_channel=1,n_class=1):
        super(ec_1dCNN, self).__init__()
        self.conv1 = nn.Conv1d(input_channel,8,5,padding=2)
        self.conv2 = nn.Conv1d(8,16,5,padding=2)
        self.conv3 = nn.Conv1d(16,32,5,padding=2)
        self.pool = nn.MaxPool1d(2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(160,64)
        self.fc2 = nn.Linear(64,32)
        self.fc3 = nn.Linear(32,n_class)
        self.conv1_bn = nn.BatchNorm1d(8)
        self.conv2_bn = nn.BatchNorm1d(16)
        self.conv3_bn = nn.BatchNorm1d(32)
        self.fc1_bn = nn.BatchNorm1d(64)
        self.fc2_bn = nn.BatchNorm1d(32)
    
    def forward(self,x):
        x = self.pool(self.conv1_bn(F.relu(self.conv1(x))))
        x = self.pool(self.conv2_bn(F.relu(self.conv2(x))))
        x = self.pool(self.conv3_bn(F.relu(self.conv3(x))))
        x = self.flatten(x)
        x = self.fc1_bn(F.relu(self.fc1(x)))
        x = self.fc2_bn(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x