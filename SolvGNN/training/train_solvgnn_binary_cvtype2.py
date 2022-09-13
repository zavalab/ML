from __future__ import absolute_import

import sys, random, pickle, csv, time
import torch
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold

from solvgnn.util.generate_dataset_for_training import solvent_dataset_binary, collate_solvent_binary
from solvgnn.model.model_GNN import solvgnn_binary
import matplotlib.pyplot as plt

class AccumulationMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.value = 0.0
        self.sum = 0.0
        self.count = 0.0
        self.avg = 0.0

    def update(self, value, n=1):
        self.value = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count

def train(epoch,train_loader,empty_solvsys,model,loss_fn1,loss_fn2,optimizer):
    stage = "train"
    model.train()

    batch_time = AccumulationMeter()
    loss_accum = AccumulationMeter()
    loss1_accum = AccumulationMeter()
    loss2_accum = AccumulationMeter()
    end = time.time()
    
    for i, solvdata in enumerate(train_loader):
        labgam1 = solvdata['gamma1'].float().cuda()
        labgam2 = solvdata['gamma2'].float().cuda()
        output = model(solvdata,empty_solvsys)                    
        loss1 = loss_fn1(output[:,0],labgam1)
        loss2 = loss_fn2(output[:,1],labgam2)
        loss = 0.5*loss1+0.5*loss2
        loss_accum.update(loss.item(),labgam1.size(0))
        loss1_accum.update(loss1.item(), labgam1.size(0))
        loss2_accum.update(loss2.item(), labgam2.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 500 == 0:
            print('Epoch [{}][{}/{}]'
                  'Time {:.3f} ({:.3f})\t'
                  'Loss {:.2f} ({:.2f})\t'
                  'Loss1 {:.2f} ({:.2f})\t'
                  'Loss2 {:.2f} ({:.2f})\t'.format(
                epoch + 1, i, len(train_loader), 
                batch_time.value, batch_time.avg,
                loss_accum.value, loss_accum.avg,
                loss1_accum.value, loss1_accum.avg,
                loss2_accum.value, loss2_accum.avg))
            
    print("[Stage {}]: Epoch {} finished with loss={:.3f} loss1={:.3f} loss2={:.3f}".format(
            stage, epoch + 1, loss_accum.avg, loss1_accum.avg, loss2_accum.avg))
    return [loss_accum.avg, loss1_accum.avg, loss2_accum.avg]

def validate(epoch,val_loader,empty_solvsys,model,loss_fn1,loss_fn2):
    stage = 'validate'
    batch_time = AccumulationMeter()
    loss_accum = AccumulationMeter()
    loss1_accum = AccumulationMeter()
    loss2_accum = AccumulationMeter()
    model.eval()
    with torch.set_grad_enabled(False):
        end = time.time()
        for i, solvdata in enumerate(val_loader):
            labgam1 = solvdata['gamma1'].float().cuda()
            labgam2 = solvdata['gamma2'].float().cuda()
            output = model(solvdata,empty_solvsys)                    
            loss1 = loss_fn1(output[:,0],labgam1)
            loss2 = loss_fn2(output[:,1],labgam2)
            loss = 0.5*loss1+0.5*loss2
            loss_accum.update(loss.item(),labgam1.size(0))
            loss1_accum.update(loss1.item(), labgam1.size(0))
            loss2_accum.update(loss2.item(), labgam2.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 500 == 0:
                print('Epoch [{}][{}/{}]'
                      'Time {:.3f} ({:.3f})\t'
                      'Loss {:.2f} ({:.2f})\t'
                      'Loss1 {:.2f} ({:.2f})\t'
                      'Loss2 {:.2f} ({:.2f})\t'.format(
                    epoch + 1, i, len(val_loader), 
                    batch_time.value, batch_time.avg,
                    loss_accum.value, loss_accum.avg,
                    loss1_accum.value, loss1_accum.avg,
                    loss2_accum.value, loss2_accum.avg))

    print("[Stage {}]: Epoch {} finished with loss={:.3f} loss1={:.3f} loss2={:.3f}".format(
            stage, epoch + 1, loss_accum.avg, loss1_accum.avg, loss2_accum.avg))
    return [loss_accum.avg, loss1_accum.avg, loss2_accum.avg]


def main():

    all_start = time.time()

    log_file = '../saved_model/print.log'
    sys.stdout = open(log_file, "w")
    # fix seed
    seed = 2021
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # read dataset file
    dataset_path = './solvgnn/data/output_binary_with_inf_all.csv'
    solvent_list_path = './solvgnn/data/solvent_list.csv'
    dataset = solvent_dataset_binary(
        input_file_path=dataset_path,
        solvent_list_path=solvent_list_path,
        generate_all=True)
    tpsa_binary = dataset.dataset['tpsa_binary_avg'].to_numpy()
    dataset_size = len(dataset)
    all_ind = np.arange(dataset_size)
    
    # print dataset size
    print('dataset size: {}'.format(dataset_size))
    
    
    
    cv_index = 0
    index_list_train = []
    index_list_valid = []
    
    for int_type in range(3):
        valid_indices = np.where(tpsa_binary==int_type)[0]
        train_indices = [idx for idx in all_ind if idx not in valid_indices]
        index_list_train.append(train_indices)
        index_list_valid.append(valid_indices)
        
        # initialize model
        model = solvgnn_binary(in_dim=74, hidden_dim=256, n_classes=1).cuda()
        model_arch = 'solvgnn'
        loss_fn1 = nn.MSELoss().cuda()
        loss_fn2 = nn.MSELoss().cuda()
        optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
        batch_size = 100
        
        # load dataset
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(valid_indices)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                   sampler=train_sampler,
                                                   collate_fn=collate_solvent_binary,
                                                   shuffle=False,
                                                   drop_last=True)
        val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                 sampler=valid_sampler,
                                                 collate_fn=collate_solvent_binary,
                                                 shuffle=False,
                                                 drop_last=True)
        empty_solvsys = dataset.generate_solvsys(batch_size)
        
        best_loss = 1000000
        train_loss_save = []
        train_loss1_save = []
        train_loss2_save = []
        val_loss_save = []
        val_loss1_save = []
        val_loss2_save = []
        
        
        for epoch in range(100):
                
            train_loss = train(epoch,train_loader,empty_solvsys,model,loss_fn1,loss_fn2,optimizer)
            train_loss_save.append(train_loss[0])
            train_loss1_save.append(train_loss[1])
            train_loss2_save.append(train_loss[2])
            val_loss = validate(epoch,val_loader,empty_solvsys,model,loss_fn1,loss_fn2)
            val_loss_save.append(val_loss[0])
            val_loss1_save.append(val_loss[1])
            val_loss2_save.append(val_loss[2])
            
            is_best = val_loss[0] < best_loss
            best_loss = min(val_loss[0], best_loss)
            if is_best:
                torch.save({
                        'epoch': epoch + 1,
                        'model_arch': model_arch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_loss': best_loss
                        }, '../saved_model/best_val_model_cv{}.pth'.format(cv_index))
            #, _use_new_zipfile_serialization=False)
        torch.save({
                'epoch': epoch + 1,
                'model_arch': model_arch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss
                }, '../saved_model/final_model_cv{}.pth'.format(cv_index))
        #, _use_new_zipfile_serialization=False)    
        
        np.save('../saved_model/train_loss_cv{}.npy'.format(cv_index),np.array(train_loss_save))
        np.save('../saved_model/train_loss1_cv{}.npy'.format(cv_index),np.array(train_loss1_save))
        np.save('../saved_model/train_loss2_cv{}.npy'.format(cv_index),np.array(train_loss2_save))
        np.save('../saved_model/val_loss_cv{}.npy'.format(cv_index),np.array(val_loss_save))
        np.save('../saved_model/val_loss1_cv{}.npy'.format(cv_index),np.array(val_loss1_save))
        np.save('../saved_model/val_loss2_cv{}.npy'.format(cv_index),np.array(val_loss2_save))
    
        cv_index += 1
    
    np.save('../saved_model/train_ind_list.npy',index_list_train)
    np.save('../saved_model/valid_ind_list.npy',index_list_valid)
     
       
    train_mse = []
    valid_mse = []    
    plt.figure(figsize=(16,8))
    for cv_index in range(3):
        train_losses = np.load('../saved_model/train_loss_cv{}.npy'.format(cv_index))
        valid_losses = np.load('../saved_model/val_loss_cv{}.npy'.format(cv_index))
        plt.subplot(2,3,cv_index+1)
        plt.plot(train_losses,label="train loss cv{}".format(cv_index+1))
        plt.plot(valid_losses,label="valid loss cv{}".format(cv_index+1))
        plt.xlabel("epoch (training iteration)")
        plt.ylabel("loss")
        plt.legend(loc="best")
        train_mse.append(train_losses[epoch])
        valid_mse.append(valid_losses[epoch])
    train_mse = np.sqrt(np.array(train_mse))
    valid_mse = np.sqrt(np.array(valid_mse))
    rmse_str = (r'Train RMSE = {:.2f} $\pm$ {:.2f}'
                '\n'
                r'Val RMSE = {:.2f} $\pm$ {:.2f}'.format(
            np.mean(train_mse), np.std(train_mse), np.mean(valid_mse), np.std(valid_mse)))
    plt.subplot(2,3,6)
    plt.text(0,0.5, rmse_str, fontsize=12)
    plt.axis('off')
    plt.savefig('../saved_model/cvloss.png',dpi=300)       
    
    all_end = time.time() - all_start
    print(all_end)


if __name__ == '__main__':
    main()