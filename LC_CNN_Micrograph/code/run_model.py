# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 18:25:02 2021

@author: sqin34
"""

from __future__ import absolute_import

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,cross_val_score,KFold,StratifiedKFold
from sklearn.metrics import r2_score,mean_squared_error

from generate_dataset import LCDataset, ToTensor
from CNN_models import CNNNet_BN_multiclass
import torch.backends.cudnn as cudnn

def main():
    
    img_path = "../"
    dataset = LCDataset(metadata=img_path+"metadata.csv",img_dir=img_path+"data/",
                        transform=Compose([ToTensor()]),
			color_space='RGB', channel=[0,1,2], read_label=True)
    train_size = dataset.__len__()
    print(train_size)
    
    seed = 2024
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    kf = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
    cv_index = 0
    index_list_train = []
    index_list_valid = []
    y = dataset.getlabelarray()
    aug_fold = 20
    original_index = np.arange(0,train_size,aug_fold)
    nSamples = np.array([130,101,101,122,104,96,93,99,99])
    normedWeights = 1 - (nSamples / np.sum(nSamples))
    normedWeights = torch.FloatTensor(normedWeights).cuda()
    valid_accuracy = []
    
    def generate_aug_indices(indices_orig,aug_fold=20):
        indices_aug = []
        for i in indices_orig:
            indices_aug.extend(np.arange(i*aug_fold,(i+1)*aug_fold))
        return indices_aug
    
    for train_indices, valid_indices in list(kf.split(original_index,y[original_index])):
        train_indices = generate_aug_indices(train_indices,aug_fold=aug_fold)
        valid_indices = valid_indices*aug_fold
        index_list_train.append(train_indices)
        index_list_valid.append(valid_indices)
    
        model = CNNNet_BN_multiclass(n_class=len(nSamples),n_channel=len(dataset.channel)).cuda()

        criterion = nn.CrossEntropyLoss(weight=normedWeights) # for classification
        optimizer = optim.Adam(model.parameters(),lr=0.0001)
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(valid_indices)
        trainloader = DataLoader(dataset, batch_size=200,
                                 sampler=train_sampler,
                                 shuffle=False,
                                 drop_last=False)
        valloader = DataLoader(dataset, batch_size=200,
                               sampler=val_sampler,
                               shuffle=False,
                               drop_last=False)
        
        train_losses = []
        valid_losses = []   
        model_path_train = 'train_cv{}'.format(cv_index+1)
        model_path_valid = 'valid_cv{}'.format(cv_index+1)
        best_train_mse = 1000000  
        best_valid_mse = 1000000
        
        for epoch in range(50):
            train_loss = 0.0
            valid_loss = 0.0
            valid_corr_count = 0
            valid_data_count = 0
        
            model.train()
            for i, data in enumerate(trainloader):
                img = data['image'].cuda()
                lab = data['label'].view(-1,1).cuda()
                lab = lab.long().squeeze()
                optimizer.zero_grad()
                output = model(img)
                loss = criterion(output, lab)
                loss.backward()
                optimizer.step()
                train_loss += loss.detach().item()
                torch.cuda.synchronize()
            train_loss /= (i + 1)
            if best_train_mse > train_loss:
                best_train_mse = train_loss
                torch.save(model.state_dict(), '{}saved_model/{}.pth'.format(img_path,model_path_train))
        
            model.eval()
            with torch.no_grad():
                for i, data in enumerate(valloader):
                    img = data['image'].cuda()
                    lab = data['label'].view(-1,1).cuda()
                    lab = lab.long().squeeze()
                    
                    output = model(img)
                    val_loss = criterion(output, lab)
                    _, pred = torch.max(output,1)
                    valid_corr_count += torch.sum(pred==lab).detach().item()
                    valid_data_count += lab.shape[0]
                    valid_loss += val_loss.detach().item()
                valid_acc = valid_corr_count / valid_data_count
                valid_loss /= (i + 1)
                if best_valid_mse > valid_loss:
                    best_valid_mse = valid_loss
                    torch.save(model.state_dict(), '{}saved_model/{}.pth'.format(img_path,model_path_valid))    
    
            print('CV {}, Epoch {}, train_loss {:.3e}, valid_loss {:.3e}'.format(cv_index+1, epoch, train_loss, valid_loss))
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
        valid_accuracy.append(valid_acc)
        np.save('{}saved_model/{}_loss.npy'.format(img_path,model_path_train),train_losses)
        np.save('{}saved_model/{}_loss.npy'.format(img_path,model_path_valid),valid_losses)
        np.save('{}saved_model/{}_ind.npy'.format(img_path,model_path_train),np.array(train_indices))
        np.save('{}saved_model/{}_ind.npy'.format(img_path,model_path_valid),np.array(valid_indices))
        np.save('{}saved_model/{}_acc.npy'.format(img_path,model_path_valid),np.array(valid_accuracy))
        torch.save(model.state_dict(), '{}saved_model/{}_ep{:2d}.pth'.format(img_path,model_path_train,epoch+1))
        cv_index += 1

    np.save('{}saved_model/{}_valid_accuracy.npy'.format(img_path,model_path_valid),np.array(valid_accuracy))        
    np.save('{}saved_model/{}_ind_list_train.npy'.format(img_path,model_path_valid),index_list_train)
    np.save('{}saved_model/{}_ind_list_valid.npy'.format(img_path,model_path_valid),index_list_valid)


if __name__ == '__main__':
    main()