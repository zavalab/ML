import pickle
import torch
from PIL import __version__ as PILLOW_VERSION
from torch.utils.data import Subset
from torchvision.transforms import Compose
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import torch.nn as nn
from generate_dataset import LCDataset, ToTensor
from CNN_models import LCCNN_AE_Classifier
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score


cv_accuracy = np.load('../saved_model/classifier_valid_accuracy.npy')
print(cv_accuracy)
output_df_all = pd.DataFrame()
n_class = 12
n_channel = 3

ind_list_train = np.load("../saved_model/classifier_ind_list_train.npy", allow_pickle=True)
ind_list_valid = np.load("../saved_model/classifier_ind_list_valid.npy", allow_pickle=True)

for cv_id in np.arange(5):
    train_ind = ind_list_train[cv_id]
    valid_ind = ind_list_valid[cv_id]

    model = LCCNN_AE_Classifier(n_channel=n_channel, n_class=n_class).cuda()
    checkpoint = torch.load('../saved_model/model_cv{}_ep50.pth'.format(cv_id+1))
    model.load_state_dict(checkpoint)

    dataset = LCDataset(metadata="../metadata.csv",img_dir="../../../data/all_pfas_droplets/",
                        transform=Compose([ToTensor()]),
                        color_space='RGB', channel=[0,1,2], read_label=True)
    valid_dataset = Subset(dataset,valid_ind)
    
    img_fn = list(dataset.metadata['img_fn'])
    testloader = DataLoader(valid_dataset, batch_size=1,
                            shuffle=False)
    output_df = dataset.metadata.iloc[valid_ind].copy()
    model.eval()
    trues = []
    preds = []
    train_val = []
    with torch.no_grad():
        for i, data in enumerate(testloader):
            img = data['image'].cuda()
            lab = data['label'].view(-1,1).cuda()
            lab = lab[0].long()
            recon, latent, class_output = model(img)
            _, pred = torch.max(class_output,1)
            preds.append(pred.item())
            trues.append(lab.item())
            train_val.append("valid")
            if i % 50 == 0:
                print('{} out of {} done!'.format(i, len(testloader)))
    preds = np.array(preds)
    trues = np.array(trues)
    output_df['true_label'] = trues
    output_df['pred_label'] = preds
    output_df['train_val'] = train_val
    output_df_all = pd.concat([output_df_all,output_df])
    print("accuracy rate {:2f}%".format(np.sum(preds==trues)/len(trues)*100))
print("overall accuracy rate {:2f}%".format(np.sum(output_df_all["true_label"].to_numpy()==output_df_all["pred_label"].to_numpy())/len(output_df_all)*100))

output_df_all['error'] = output_df_all.pred_label == output_df_all.true_label
output_df_all.to_csv("../analysis/output_df_all5valid.csv",index=False)