# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 18:40:55 2021

@author: sqin34
"""

import numpy as np

def get_rdp_center(img_gray):
    mask = np.array(img_gray >= np.quantile(img_gray,0.7)).astype('int')
    mask = mask / np.sum(mask)

    dx = np.sum(mask, 0)
    dy = np.sum(mask, 1)
    
    cx = np.round(np.sum(dx * np.arange(img_gray.shape[1])))
    cy = np.round(np.sum(dy * np.arange(img_gray.shape[0])))
    
    center = np.array([cx,cy]).astype('int')
    
    return mask, center

def get_rdp_distance(img_gray,center):
    pixel_ind = np.indices((img_gray.shape))
    pixel_dist = pixel_ind - center[:,np.newaxis,np.newaxis]
    pixel_dist = np.round(np.sqrt(np.sum(pixel_dist ** 2, axis=0))).astype('int')
#    plt.imshow(pixel_dist,cmap='gray')
    dist_min, dist_max = np.min(pixel_dist), np.max(pixel_dist)
    return dist_min, dist_max, pixel_dist

def get_rdp_profile(dist_min,dist_max,pixel_dist,img):
    intensity_count = np.zeros((dist_max+1)).astype('int')
    intensity_sum = np.zeros((dist_max+1))    
    for i in range(dist_max+1):
        intensity_sum[i] = np.sum(img[pixel_dist == i])
        intensity_count[i] = np.sum(pixel_dist == i)
    intensity_avg = intensity_sum/intensity_count
    return intensity_avg

def create_rdp_padding(rdp_arr):
    rdp_padded = np.zeros([len(rdp_arr),len(max(rdp_arr,key = lambda x: len(x)))])
    for i,j in enumerate(rdp_arr):
        rdp_padded[i][0:len(j)] = j
    return rdp_padded