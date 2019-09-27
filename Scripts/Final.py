# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 18:08:56 2019

@author: DELL
"""
import cv2
import os
import numpy as np
import pandas as pd
from scipy import signal
from matplotlib import pyplot as plt
from skimage import data

Data= os.listdir('C:\\Users\\DELL\\Desktop\\Glaucoma-Detection\\Glaucoma-Fundus')
for i in range(30):
    img = cv2.imread('C:\\Users\\DELL\\Desktop\\Glaucoma-Detection\\Glaucoma-Fundus' + "/" + Data[i])
    R,G,B= cv2.split(img)  
    
    #splitting into 3 channels
    
    RC = R-R.mean()#Preprocessing Red
    RC = RC-RC.mean()-R.std() #Preprocessing Red
    RC = RC-RC.mean()-RC.std() #Preprocessing Red
    
    MeanR = RC.mean()#Mean of preprocessed red
    SDR = RC.std()#SD of preprocessed red
    
    # Thr = 49.5 - 12 - Ar.std()               
    #OD Threshold
    
    Thr = RC.std()
    print(Thr)
    
    GC = G - G.mean()#Preprocessing Green
    GC= GC- GC.mean()-G.std() #Preprocessing Green
    
    MeanG = GC.mean()#Mean of preprocessed green
    SDG = GC.std()#SD of preprocessed green
    Thg = GC.mean() + 2*GC.std() + 49.5 + 12 #OC Threshold
    
    filter = signal.gaussian(99, std=6) #Gaussian Window
    filter=filter/sum(filter)
    
    hist,bins = np.histogram(GC.ravel(),256,[0,256])#Histogram of preprocessed green channel
    histr,binsr = np.histogram(RC.ravel(),256,[0,256])#Histogram of preprocessed red channel
    
    smooth_hist_g=np.convolve(filter,hist)  #Histogram Smoothing Green
    smooth_hist_r=np.convolve(filter,histr) #Histogram Smoothing Red
    
    plt.subplot(2, 2, 1)
    plt.plot(hist)
    plt.title("Preprocessed Green Channel")
    
    plt.subplot(2, 2, 2)
    plt.plot(smooth_hist_g)
    plt.title("Smoothed Histogram Green Channel")
    
    plt.subplot(2, 2, 3)
    plt.plot(histr)
    plt.title("Preprocessed Red Channel")
    
    plt.subplot(2, 2, 4)
    plt.plot(smooth_hist_r)
    plt.title("Smoothed Histogram Red Channel")
    
    plt.show()
    
    
    
    r,c = GC.shape
    Dd = np.zeros(shape=(r,c))
    Dc = np.zeros(shape=(r,c))
    
    for i in range(1,r):
    	for j in range(1,c):
    		if RC[i,j]>Thr:
    			Dd[i,j]=255
    		else:
    			Dd[i,j]=0
    
    for i in range(1,r):
    	for j in range(1,c):
    		if GC[i,j]>Thg:
    			Dc[i,j]=1
    		else:
    			Dc[i,j]=0
    
    
    
    plt.imshow(Dd, cmap = 'gray', interpolation = 'bicubic')
    plt.axis("off")
    plt.title("Optic Disk")
    plt.show()
    
    plt.imshow(Dc, cmap = 'gray', interpolation = 'bicubic')
    plt.axis("off")
    plt.title("Optic Cup")
    plt.show()	