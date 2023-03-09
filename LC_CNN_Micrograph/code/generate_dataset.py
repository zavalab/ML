# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 15:47:57 2021

@author: sqin34
"""

from __future__ import absolute_import

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2


class LCDataset(Dataset):
    def __init__(self, metadata, img_dir, transform=None,
                 color_space = 'RGB',
                 channel = [0,1,2], read_label = True):

        self.metadata = pd.read_csv(metadata)
        self.img_dir = img_dir
        self.transform = transform
        self.color_space = color_space
        self.channel = channel
        self.read_label = read_label

    def __len__(self):
        return len(self.metadata)
    
    def getlabelarray(self):
        return self.metadata.label.to_numpy()
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.img_dir + self.metadata.iloc[idx, 0]
        image = cv2.imread(img_name)
        if self.color_space == 'RGB':            
            image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            image = image/255
            image = image[:,:,self.channel]
        elif self.color_space == 'LAB':
            image = cv2.cvtColor(image,cv2.COLOR_BGR2LAB)
            image = image * [100/255,1,1]-[0,128,128]
            image = image[:,:,self.channel]
        elif self.color_space == 'gray':
            image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            image = image/255
            image = image[:,:,np.newaxis]
        image = image.astype('float32')
        if self.read_label:
            label = self.metadata.iloc[idx, 1:]
            label = np.array([label])
            label = label.astype('float32').reshape(-1, 1)
        else:
            label = np.array([None],dtype='float32')
        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)
        
        return sample


class ToTensor(object):

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = image.transpose((2, 0, 1))
        return {'image': torch.tensor(image),
                'label': torch.tensor(label)}
