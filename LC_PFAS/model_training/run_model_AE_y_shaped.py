# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 18:25:02 2021

@author: sqin34
"""

from __future__ import absolute_import

import time
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
from CNN_models import LCCNN_AE_Classifier
import torch.backends.cudnn as cudnn

def main():
        
    preload = False
    preload_epoch = 0
    if preload is True:
        assert preload_epoch > 0

    img_path = "../"
    dataset = LCDataset(metadata=img_path+"metadata.csv",img_dir=img_path+"../../data/all_pfas_droplets/",
                        transform=Compose([ToTensor()]),
			color_space='RGB', channel=[0,1,2], read_label=True)
    train_size = dataset.__len__()
    print(train_size)
    
    seed = 2023
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    cv_index = 0
    kf = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
    index_list_train = []
    index_list_valid = []
    y = dataset.getlabelarray()
    aug_fold = 20
    original_index = np.arange(0,train_size,aug_fold)
    nSamples = dataset.metadata.groupby("label")["label"].count().to_numpy() // aug_fold
    print(len(nSamples))
    print(nSamples)
    normedWeights = 1 - (nSamples / np.sum(nSamples))
    normedWeights = torch.FloatTensor(normedWeights).cuda()    

    def generate_aug_indices(indices_orig,aug_fold=20):
        indices_aug = []
        for i in indices_orig:
            indices_aug.extend(np.arange(i*aug_fold,(i+1)*aug_fold))
        return indices_aug

    alpha = 0.9
    weight_recon = 0.5
    weight_classifier = 0.5
    scale_factor_recon = 1.

    for train_indices, valid_indices in list(kf.split(original_index,y[original_index])):
        train_indices = generate_aug_indices(train_indices,aug_fold=aug_fold)
        valid_indices = valid_indices*aug_fold
        index_list_train.append(train_indices)
        index_list_valid.append(valid_indices)
        model = LCCNN_AE_Classifier(n_channel=len(dataset.channel), n_class=len(nSamples)).cuda()
        if preload is True:
            checkpoint = torch.load('{}saved_model/train_model_ep{}.pth'.format(img_path, preload_epoch))
            model.load_state_dict(checkpoint)
            train_losses_recon = list(np.load('{}saved_model/train_loss_recon_ep{}.npy'.format(img_path, preload_epoch)))
            train_losses_classifier = list(np.load('{}saved_model/train_loss_classifier_ep{}.npy'.format(img_path, preload_epoch)))
            train_losses_total = list(np.load('{}saved_model/train_loss_total_ep{}.npy'.format(img_path, preload_epoch)))
        else:
            train_losses_recon = []
            train_losses_classifier = []
            train_losses_total = []

        criterion_recon = nn.MSELoss()
        criterion_classifier = nn.CrossEntropyLoss(normedWeights)
        optimizer = optim.Adam(model.parameters(),lr=0.001)

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

        valid_losses_recon = []
        valid_losses_classifier = []
        valid_losses_total = []
        valid_accuracies = []
        best_valid_loss_total = 1000000
        
        avg_loss_recon = 0.001
        avg_loss_classifier = 0.001

        for epoch in range(50):
            train_loss_recon = 0.0
            train_loss_classifier = 0.0
            train_loss_total = 0.0
            valid_loss_recon = 0.0
            valid_loss_classifier = 0.0
            valid_loss_total = 0.0
            valid_corr_count = 0.0
            valid_data_count = 0.0
            start_time = time.time()
            model.train()
            for i, data in enumerate(trainloader):
                optimizer.zero_grad()
                img = data['image'].cuda()
                lab = data['label'].view(-1,1).cuda()
                lab = lab.long().squeeze()
                recon, _, class_output = model(img)
                loss_recon = scale_factor_recon * criterion_recon(recon, img)
                loss_classifier = criterion_classifier(class_output, lab)

                avg_loss_recon = alpha * avg_loss_recon + (1 - alpha) * loss_recon.item()
                avg_loss_classifier = alpha * avg_loss_classifier + (1 - alpha) * loss_classifier.item()
                norm_train_loss_recon = loss_recon / avg_loss_recon
                norm_train_loss_classifier = loss_classifier / avg_loss_classifier

                loss_total = weight_recon * norm_train_loss_recon + weight_classifier * norm_train_loss_classifier
                loss_total.backward()
                optimizer.step()
                train_loss_recon += loss_recon.detach().item()
                train_loss_classifier += loss_classifier.detach().item()
                train_loss_total += loss_total.detach().item()
            train_loss_recon = train_loss_recon / (i+1)
            train_loss_classifier = train_loss_classifier / (i+1)
            train_loss_total = train_loss_total / (i+1)
            train_losses_recon.append(train_loss_recon)
            train_losses_classifier.append(train_loss_classifier)
            train_losses_total.append(train_loss_total)
            train_elapsed_time = int((time.time() - start_time) / 60)
            
            start_time = time.time()
            model.eval()
            with torch.no_grad():
                for i, data in enumerate(valloader):
                    img = data['image'].cuda()
                    lab = data['label'].view(-1,1).cuda()
                    lab = lab.long().squeeze()
                    recon, latent, class_output = model(img)
                    loss_recon = scale_factor_recon * criterion_recon(recon, img)
                    loss_classifier = criterion_classifier(class_output, lab)

                    norm_valid_loss_recon = loss_recon / avg_loss_recon
                    norm_valid_loss_classifier = loss_classifier / avg_loss_classifier

                    loss_total = weight_recon * norm_valid_loss_recon + weight_classifier * norm_valid_loss_classifier
                    _, pred = torch.max(class_output,1)
                    valid_corr_count += torch.sum(pred==lab).detach().item()
                    valid_data_count += lab.shape[0]
                    valid_loss_recon += loss_recon.detach().item()
                    valid_loss_classifier += loss_classifier.detach().item()
                    valid_loss_total += loss_total.detach().item()
                valid_acc = valid_corr_count / valid_data_count
                valid_loss_recon = valid_loss_recon / (i+1)
                valid_loss_classifier = valid_loss_classifier / (i+1)
                valid_loss_total = valid_loss_total / (i+1)
                valid_losses_recon.append(valid_loss_recon)
                valid_losses_classifier.append(valid_loss_classifier)
                valid_losses_total.append(valid_loss_total)
                if best_valid_loss_total > valid_loss_total:
                    best_valid_loss_total = valid_loss_total
                    torch.save(model.state_dict(), '{}saved_model/model_cv{}_bestval.pth'.format(img_path,cv_index+1))
                valid_elapsed_time = int((time.time() - start_time) / 60)
            print('CV {}, Epoch {}, train_loss {:.3e}, etime {} min, val_loss {:.3e}, etime {} min'.format(
                cv_index+1, epoch+1, train_loss_total, train_elapsed_time, valid_loss_total, valid_elapsed_time))
        print("End of cv{}, Accuracy {}".format(cv_index+1,valid_acc))
        valid_accuracies.append(valid_acc)
        np.save('{}saved_model/model_cv{}_trainloss_recon.npy'.format(img_path,cv_index+1),train_losses_recon)
        np.save('{}saved_model/model_cv{}_trainloss_classifier.npy'.format(img_path,cv_index+1),train_losses_classifier)
        np.save('{}saved_model/model_cv{}_trainloss_total.npy'.format(img_path,cv_index+1),train_losses_total)
        np.save('{}saved_model/model_cv{}_validloss_recon.npy'.format(img_path,cv_index+1),valid_losses_recon)
        np.save('{}saved_model/model_cv{}_validloss_classifier.npy'.format(img_path,cv_index+1),valid_losses_classifier)
        np.save('{}saved_model/model_cv{}_validloss_total.npy'.format(img_path,cv_index+1),valid_losses_total)
        np.save('{}saved_model/model_cv{}_train_ind.npy'.format(img_path,cv_index+1),np.array(train_indices))
        np.save('{}saved_model/model_cv{}_valid_ind.npy'.format(img_path,cv_index+1),np.array(valid_indices))
        torch.save(model.state_dict(), '{}saved_model/model_cv{}_ep{:2d}.pth'.format(img_path,cv_index+1,epoch+1))
        cv_index += 1
    np.save('{}saved_model/classifier_valid_accuracy.npy'.format(img_path),np.array(valid_accuracies))
    np.save('{}saved_model/classifier_ind_list_train.npy'.format(img_path),index_list_train)
    np.save('{}saved_model/classifier_ind_list_valid.npy'.format(img_path),index_list_valid)



if __name__ == '__main__':
    main()