# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 11:20:47 2020

@author: sqin34
"""

import matplotlib.pyplot as plt
import numpy as np
import cv2
import imutils,re
from imutils import contours,perspective
from scipy.spatial import distance as dist


def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


def BBox_mask(image,mode,size_thres=100,filename=None,replace=False):
    mode_list = ["density","bbox_rec","bbox_rec_largest","crop","crop_largest",
                 "crop_largest_from_raw","crop_largest_from_raw_bf","crop_largest_from_raw_bflc",
                 "all","draw"]
    if mode not in mode_list:
        print("Invalid mode. Please input valid mode from list {}".format(mode_list))
        return
    image_transform = image.copy()

    if "raw" in mode:
        if "bf" in mode:
            image_transform = cv2.GaussianBlur(image_transform,(13,13),0)
            image_transform = cv2.Canny(image_transform,55,60)
            image_transform = cv2.dilate(image_transform,np.ones((9,9)),iterations=3)
        elif "bflc" in mode:
            image_transform = cv2.GaussianBlur(image_transform,(21,21),0) 
            image_transform = cv2.Canny(image_transform,30,30)
            image_transform = cv2.dilate(image_transform,np.ones((9,9)),iterations=3)
            
        else:
            image_transform = cv2.GaussianBlur(image_transform,(9,9),0)
            image_transform = cv2.Canny(image_transform,20,50)
            image_transform = cv2.dilate(image_transform,np.ones((9,9)),iterations= 10)
    else:
        image_transform = cv2.GaussianBlur(image_transform,(7,7),0)
        image_transform = cv2.Canny(image_transform,10,18)
        image_transform = cv2.dilate(image_transform,None,iterations=1)

    cnts = cv2.findContours(image_transform.copy(), cv2.RETR_EXTERNAL,
                           cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    (cnts, _) = contours.sort_contours(cnts)
    
    if mode == "density":
        num_droplet = 0
        rec_box_area = []
        for c in cnts:
            if cv2.contourArea(c) < size_thres:
                continue
            num_droplet += 1
            box = cv2.boundingRect(c)
            rec_box_area.append(box[2]*box[3])
        return num_droplet,rec_box_area
    
    if mode == "bbox_rec":
        rec_box = np.zeros((image.shape[0],image.shape[1])).astype(int)
        for c in cnts:
            if cv2.contourArea(c) < size_thres:
                continue
            box = cv2.boundingRect(c)
            box = ((box[0],box[1]),(box[0]+box[2],box[1]),(box[0]+box[2],box[1]+box[3]),(box[0],box[1]+box[3]))
            box = np.array(box, dtype="int")
            box = perspective.order_points(box)
            box = np.array(box, dtype="int")
            (tl, tr, br, bl) = box
            rec_box[tl[1]:br[1],tl[0]:br[0]] = 1
        rec_box = np.stack((rec_box,rec_box,rec_box),axis=2)
        return rec_box
    
    if mode == "bbox_rec_largest":

        rec_box = np.zeros((image.shape[0],image.shape[1])).astype(int)
        cnts = sorted(cnts, key=lambda x: cv2.contourArea(x))
        c = cnts[-1]
        box = cv2.boundingRect(c)
        box = ((box[0],box[1]),(box[0]+box[2],box[1]),(box[0]+box[2],box[1]+box[3]),(box[0],box[1]+box[3]))
        box = np.array(box, dtype="int")
        box = perspective.order_points(box)
        box = np.array(box, dtype="int")
        (tl, tr, br, bl) = box
        rec_box[tl[1]:br[1],tl[0]:br[0]] = 1
        rec_box = np.stack((rec_box,rec_box,rec_box),axis=2)
        return rec_box
    
    if mode == "crop":
        num_droplet = 0
        for c in cnts:
            if cv2.contourArea(c) < size_thres:
                continue
            num_droplet += 1
            box = cv2.boundingRect(c)
            box = ((box[0],box[1]),(box[0]+box[2],box[1]),(box[0]+box[2],box[1]+box[3]),(box[0],box[1]+box[3]))
            box = np.array(box, dtype="int")
            box = perspective.order_points(box)
            box = np.array(box, dtype="int")
            (tl, tr, br, bl) = box
            crop_img = image[tl[1]:br[1],tl[0]:br[0],:]
            crop_name = re.sub(".jpg","_crop{:02d}.jpg".format(num_droplet),filename)
            cv2.imwrite(crop_name,crop_img)
        return        

    if mode == "crop_largest":
        cnts = sorted(cnts, key=lambda x: cv2.contourArea(x))
        c = cnts[-1]
        box = cv2.boundingRect(c)
        box = ((box[0],box[1]),(box[0]+box[2],box[1]),(box[0]+box[2],box[1]+box[3]),(box[0],box[1]+box[3]))
        box = np.array(box, dtype="int")
        box = perspective.order_points(box)
        box = np.array(box, dtype="int")
        (tl, tr, br, bl) = box
        crop_img = image[tl[1]:br[1],tl[0]:br[0],:]
        crop_name = re.sub(".jpg","_cropL.jpg",filename)
        cv2.imwrite(crop_name,crop_img)
        return
    
    if "crop_largest_from_raw" in mode:
        cnts = sorted(cnts, key=lambda x: cv2.contourArea(x))
        c = cnts[-1]
        box = cv2.boundingRect(c)
        width = round(box[2]/2)
        height = round(box[3]/2)
        center = (box[0]+width,box[1]+height)
        box_length = max(width,height)
        box = ((center[0]-box_length,center[1]-box_length), #top left
               (center[0]+box_length,center[1]-box_length), #top right
               (center[0]+box_length,center[1]+box_length), #bottom right
               (center[0]-box_length,center[1]+box_length)) #bottom left
        box = np.array(box, dtype="int")
        box = perspective.order_points(box)
        box = np.array(box, dtype="int")
        (tl, tr, br, bl) = box
        crop_img = image[tl[1]:br[1],tl[0]:br[0],:]
        crop_name = re.sub(".jpg","_cropL.jpg",filename)
        if 0 in crop_img.shape:
            print("Manual Cropping needed: " + crop_name)
            return 0
        else:
            if replace == True:
                cv2.imwrite(filename,crop_img)
            else:
                cv2.imwrite(crop_name,crop_img)
            return box_length*2
    
    if mode == "all":
        num_droplet = 0
        rec_box = np.zeros((image.shape[0],image.shape[1])).astype(int)

        for c in cnts:
            if cv2.contourArea(c) < size_thres:
                continue
            num_droplet += 1
            box = cv2.boundingRect(c)
            box = ((box[0],box[1]),(box[0]+box[2],box[1]),(box[0]+box[2],box[1]+box[3]),(box[0],box[1]+box[3]))
            box = np.array(box, dtype="int")
            box = perspective.order_points(box)
            box = np.array(box, dtype="int")
            (tl, tr, br, bl) = box
            rec_box[tl[1]:br[1],tl[0]:br[0]] = 1
        rec_box = np.stack((rec_box,rec_box,rec_box),axis=2)
        return num_droplet, rec_box_area, rec_box        

    if mode == "draw":
        for c in cnts:
            if cv2.contourArea(c) < size_thres:
                continue
            image_original = image.copy()
            box = cv2.boundingRect(c)
            box = ((box[0],box[1]),(box[0]+box[2],box[1]),(box[0]+box[2],box[1]+box[3]),(box[0],box[1]+box[3]))
            box = np.array(box, dtype="int")
            box = perspective.order_points(box)
            box = np.array(box, dtype="int")
            cv2.drawContours(image_original, [box], -1, (0, 255, 0), 1)
            for (x, y) in box:
                cv2.circle(image_original, (x, y), 2, (0, 0, 255), -1)
            (tl, tr, br, bl) = box
            (tltrX, tltrY) = midpoint(tl, tr)
            (blbrX, blbrY) = midpoint(bl, br)
            (tlblX, tlblY) = midpoint(tl, bl)
            (trbrX, trbrY) = midpoint(tr, br)
            cv2.circle(image_original, (int(tltrX), int(tltrY)), 2, (255, 0, 0), -1)
            cv2.circle(image_original, (int(blbrX), int(blbrY)), 2, (255, 0, 0), -1)
            cv2.circle(image_original, (int(tlblX), int(tlblY)), 2, (255, 0, 0), -1)
            cv2.circle(image_original, (int(trbrX), int(trbrY)), 2, (255, 0, 0), -1)
            cv2.line(image_original, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
            		(255, 0, 255), 1)
            cv2.line(image_original, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
            		(255, 0, 255), 1)
            dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
            dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
            pixelsPerMetric = dB / 1           
            dimA = dA / pixelsPerMetric
            dimB = dB / pixelsPerMetric
            cv2.putText(image_original, "{:.1f}in".format(dimA),
            	(int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
            	0.3, (255, 255, 255), 1)
            cv2.putText(image_original, "{:.1f}in".format(dimB),
            	(int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
            	0.3, (255, 255, 255), 1)
            cv2.imshow("Image", image_original)
            cv2.waitKey(0)
        return               
        
