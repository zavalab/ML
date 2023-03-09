# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 14:47:44 2021

@author: sqin34
"""

import numpy as np
import cv2
import torch
from rdp_utils import get_rdp_center, get_rdp_distance, get_rdp_profile


class lc_micrograph:    
    def __init__(self,img_file):
        self.img_file = img_file
        self.img = cv2.imread(img_file)
    def get_img_matrix(self,color_space='RGB'):
        if color_space == 'RGB':
            img_mat = cv2.cvtColor(self.img,cv2.COLOR_BGR2RGB)
            img_mat = img_mat/255
        elif color_space == 'LAB':
            img_mat = cv2.cvtColor(self.img,cv2.COLOR_BGR2LAB)
            img_mat = img_mat * [100/255,1,1] - [0,128,128]
        elif color_space == 'gray':
            img_mat = cv2.cvtColor(self.img,cv2.COLOR_BGR2GRAY)
            img_mat = img_mat/255
            img_mat = img_mat[:,:,np.newaxis]
        return img_mat.astype('float32')
    def get_lc_center(self):
        img_gray = self.get_img_matrix(color_space='gray').squeeze()
        mask, center = get_rdp_center(img_gray)
        return mask,center,img_gray
    def get_lc_center_dist(self):
        _, center, img_gray = self.get_lc_center()     
        dist_min, dist_max, pixel_dist = get_rdp_distance(img_gray,center)
        return dist_min,dist_max,pixel_dist,img_gray
    def get_rdp(self,color_space='RGB'):        
        img = self.get_img_matrix(color_space=color_space)      
        dist_min, dist_max, pixel_dist, img_gray = self.get_lc_center_dist()
        if color_space == 'RGB':
            rdp_data = {
                'gray': get_rdp_profile(dist_min,dist_max,pixel_dist,img_gray),
                'rgbR': get_rdp_profile(dist_min,dist_max,pixel_dist,img[:,:,0]),
                'rgbG': get_rdp_profile(dist_min,dist_max,pixel_dist,img[:,:,1]),
                'rgbB': get_rdp_profile(dist_min,dist_max,pixel_dist,img[:,:,2])}
        elif color_space == 'LAB':
            rdp_data = {
                'gray': get_rdp_profile(dist_min,dist_max,pixel_dist,img_gray),
                'labL': get_rdp_profile(dist_min,dist_max,pixel_dist,img[:,:,0]),
                'labA': get_rdp_profile(dist_min,dist_max,pixel_dist,img[:,:,1]),
                'labB': get_rdp_profile(dist_min,dist_max,pixel_dist,img[:,:,2])}            
        return rdp_data
    def get_saliency_rdp(self,saliency):
        saliency_rdp = []
        dist_min, dist_max, pixel_dist, _ = self.get_lc_center_dist()
        if len(saliency.shape) == 3:
            n_channel = saliency.shape[2]
            for channel in range(n_channel):
                saliency_rdp.append(get_rdp_profile(dist_min,dist_max,pixel_dist,saliency[:,:,channel]))
            saliency_rdp = np.array(saliency_rdp)
        else:
            saliency_rdp = get_rdp_profile(dist_min,dist_max,pixel_dist,saliency)
        return saliency_rdp
                
    def predict(self,model,color_space='RGB',gpu=True):
        img = self.get_img_matrix(color_space=color_space)
        img = torch.tensor(img.transpose(2,0,1)).unsqueeze(0)
        if gpu:
            model = model.cuda()
            img = img.cuda()
        else:
            model = model.cpu()
            img = img.cpu()
        output = model(img)
        return output.argmax().cpu().numpy()
    def predict_with_saliency(self,model,color_space='RGB',saliency_type='grad',
                              integrated = False,
                              saliency_rdp=False,gpu=True):
        img = self.get_img_matrix(color_space=color_space)
        img = torch.tensor(img.transpose(2,0,1)).unsqueeze(0)
       
        if gpu:
            model = model.cuda()
            img = img.cuda()
        else:
            model = model.cpu()
            img = img.cpu()

        if integrated:
            from captum.attr import IntegratedGradients
            output = model(img)
            output = output.argmax()
            ig = IntegratedGradients(model)
            saliency = ig.attribute(img,n_steps=50,target=output,return_convergence_delta=False)
            saliency = saliency.abs()
        else:
            img.requires_grad_()
            output = model(img)
            output = output.max()
            output.backward()
            saliency = img.grad.data.abs()
            
        if saliency_type == 'grad':
            saliency = saliency.cpu().numpy().squeeze(0).transpose((1,2,0))
            if saliency_rdp:
                dist_min, dist_max, pixel_dist, img_gray = self.get_lc_center_dist()
                if color_space == 'RGB':
                    sal_rdp_data = {
                        'salgrad_rdpR': get_rdp_profile(dist_min,dist_max,pixel_dist,saliency[:,:,0]),
                        'salgrad_rdpG': get_rdp_profile(dist_min,dist_max,pixel_dist,saliency[:,:,1]),
                        'salgrad_rdpB': get_rdp_profile(dist_min,dist_max,pixel_dist,saliency[:,:,2])}
                elif color_space == 'LAB':
                    sal_rdp_data = {
                        'salgrad_labL': get_rdp_profile(dist_min,dist_max,pixel_dist,saliency[:,:,0]),
                        'salgrad_labA': get_rdp_profile(dist_min,dist_max,pixel_dist,saliency[:,:,1]),
                        'salgrad_labB': get_rdp_profile(dist_min,dist_max,pixel_dist,saliency[:,:,2])}
                elif color_space == 'gray':
                    sal_rdp_data = {
                        'salgrad_gray': get_rdp_profile(dist_min,dist_max,pixel_dist,saliency[:,:])}                    
                return saliency,sal_rdp_data
            else:
                return saliency
        elif saliency_type == 'maxgrad':
            saliency = saliency.cpu().numpy().squeeze(0).transpose((1,2,0))
            saliency = np.max(saliency,axis=2)
            if saliency_rdp:
                dist_min, dist_max, pixel_dist, img_gray = self.get_lc_center_dist()
                sal_rdp_data = {
                    'salmaxgrad_rdp': get_rdp_profile(dist_min,dist_max,pixel_dist,saliency)}
                return saliency,sal_rdp_data
            else:
                return saliency
        elif saliency_type == 'avggrad':
            saliency = saliency.cpu().numpy().squeeze(0).transpose((1,2,0))
            saliency = np.mean(saliency,axis=2)
            if saliency_rdp:
                dist_min, dist_max, pixel_dist, img_gray = self.get_lc_center_dist()
                sal_rdp_data = {
                    'salavggrad_rdp': get_rdp_profile(dist_min,dist_max,pixel_dist,saliency)}
                return saliency,sal_rdp_data
            else:
                return saliency
        elif saliency_type == 'inputgrad':
            saliency = saliency*img.detach()
            saliency = saliency.cpu().numpy().squeeze(0).transpose((1,2,0))
            if saliency_rdp:
                dist_min, dist_max, pixel_dist, img_gray = self.get_lc_center_dist()
                if color_space == 'RGB':
                    sal_rdp_data = {
                        'salinputgrad_rdpR': get_rdp_profile(dist_min,dist_max,pixel_dist,saliency[:,:,0]),
                        'salinputgrad_rdpG': get_rdp_profile(dist_min,dist_max,pixel_dist,saliency[:,:,1]),
                        'salinputgrad_rdpB': get_rdp_profile(dist_min,dist_max,pixel_dist,saliency[:,:,2])} 
                elif color_space == 'LAB':
                    sal_rdp_data = {
                        'salinputgrad_labL': get_rdp_profile(dist_min,dist_max,pixel_dist,saliency[:,:,0]),
                        'salinputgrad_labA': get_rdp_profile(dist_min,dist_max,pixel_dist,saliency[:,:,1]),
                        'salinputgrad_labB': get_rdp_profile(dist_min,dist_max,pixel_dist,saliency[:,:,2])}    
                elif color_space == 'gray':
                    sal_rdp_data = {
                        'salinputgrad_gray': get_rdp_profile(dist_min,dist_max,pixel_dist,saliency[:,:])} 
                return saliency,sal_rdp_data
            else:
                return saliency
        elif saliency_type == 'maxinputgrad':
            saliency = saliency*img.detach()
            saliency = saliency.cpu().numpy().squeeze(0).transpose((1,2,0))
            saliency = np.max(saliency,axis=2)
            if saliency_rdp:
                dist_min, dist_max, pixel_dist, img_gray = self.get_lc_center_dist()
                sal_rdp_data = {
                    'salmaxinputgrad_rdp': get_rdp_profile(dist_min,dist_max,pixel_dist,saliency)}  
                return saliency,sal_rdp_data
            else:
                return saliency
        elif saliency_type == 'avginputgrad':
            saliency = saliency*img.detach()
            saliency = saliency.cpu().numpy().squeeze(0).transpose((1,2,0))
            saliency = np.mean(saliency,axis=2)
            if saliency_rdp:
                dist_min, dist_max, pixel_dist, img_gray = self.get_lc_center_dist()
                sal_rdp_data = {
                    'salavginputgrad_rdp': get_rdp_profile(dist_min,dist_max,pixel_dist,saliency)}  
                return saliency,sal_rdp_data
            else:
                return saliency            
    