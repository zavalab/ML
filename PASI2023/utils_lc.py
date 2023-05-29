import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import cv2
from itertools import product
from sklearn.metrics import confusion_matrix
from sklearn.utils import check_matplotlib_support
from sklearn.base import is_classifier


class CNN(nn.Module):
    def __init__(self, n_class=1, n_channel=3):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(n_channel,32,7,padding=3)
        self.pool = nn.AvgPool2d(3,3)
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32,64,7,padding=3)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64,128,7,padding=3)
        self.conv3_bn = nn.BatchNorm2d(128)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(2048,128)
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


class ConfusionMatrixDisplay:
    def __init__(self, confusion_matrix, display_labels):
        self.confusion_matrix = confusion_matrix
        self.display_labels = display_labels

    def plot(self, include_values=True, cmap='viridis',
             xticks_rotation='horizontal', values_format=None, ax=None):

        check_matplotlib_support("ConfusionMatrixDisplay.plot")
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        cm = self.confusion_matrix
        n_classes = cm.shape[0]
        self.im_ = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        self.text_ = None

        cmap_min, cmap_max = self.im_.cmap(0), self.im_.cmap(256)

        if include_values:
            self.text_ = np.empty_like(cm, dtype=object)
            if values_format is None:
                values_format = '.2g'

            thresh = (cm.max() - cm.min()) / 2.
            for i, j in product(range(n_classes), range(n_classes)):
                color = cmap_max if cm[i, j] < thresh else cmap_min
                if cm[i,j] > 0:
                    self.text_[i, j] = ax.text(j, i,
                                               format(cm[i, j], values_format).lstrip('0'),
                                               ha="center", va="center",
                                               color=color) # color
                else:
                    self.text_[i, j] = ax.text(j, i,
                                              '-',
                                               ha="center", va="center",
                                               color=color) # color                    

        ax.set(xticks=np.arange(n_classes),
               yticks=np.arange(n_classes),
               xticklabels=self.display_labels,
               yticklabels=self.display_labels,
               ylabel="True label",
               xlabel="Predicted label")

        ax.set_ylim((n_classes - 0.5, -0.5))
        plt.setp(ax.get_xticklabels(), rotation=xticks_rotation)

        self.figure_ = fig
        self.ax_ = ax
        return self


def plot_confusion_matrix(y_pred, y_true, labels=None,
                          sample_weight=None, normalize=None,
                          display_labels=None, include_values=True,
                          xticks_rotation='horizontal',
                          values_format=None,
                          cmap='viridis', ax=None):

    cm = confusion_matrix(y_true, y_pred, sample_weight=sample_weight,
                          labels=labels, normalize=normalize)

    if display_labels is None:
        if labels is None:
            display_labels = None
        else:
            display_labels = labels

    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=display_labels)
    return disp.plot(include_values=include_values,
                     cmap=cmap, ax=ax, xticks_rotation=xticks_rotation,
                     values_format=values_format)