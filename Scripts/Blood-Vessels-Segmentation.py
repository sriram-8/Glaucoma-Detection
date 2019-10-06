# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 08:19:24 2019

@author: DELL
"""

import cv2
import pandas
import numpy as np
from matplotlib import pyplot as plt
import os 
#kernel=np.ones([[1,1,1],[1,1,1],[1,1,1]])*1/9
kernel=np.ones((15,15),np.uint8)
Data= os.listdir('C:\\Users\\DELL\\Desktop\\Glaucoma-Detection\\New')
for i in range(10):
    img = cv2.imread('C:\\Users\\DELL\\Desktop\\Glaucoma-Detection\\New' + "/" + Data[i],0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16,16))
    cl1 = clahe.apply(img)
    closing = cv2.morphologyEx(cl1, cv2.MORPH_CLOSE, kernel)
    plt.imshow(closing)
    plt.show()
    cv2.imwrite('closing.jpg',closing)

    #OR
    #The imapainting Procedure can be followed in order to extract the neccessary part
    
    plt.imshow(cl1)
    plt.show()
    median=cv2.medianBlur(cl1,5)
    plt.title('Median-Blur')
    plt.imshow(median)
    plt.show()
    cv2.imwrite('Clahe.jpg',cl1)
    cv2.imwrite('Median-Blur.jpg',median) #Enhanced Image
    
    mean =cv2.blur(median,(9,9))
    sub=abs(mean-median)
    
    
    
    thresh1 = cv2.adaptiveThreshold(median, 255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY, 199, 5) 
    plt.imshow(thresh1)
    plt.show()
    cv2.imwrite('mean.jpg',thresh1)
    sub= abs(median-thresh1);
    dst = cv2.inpaint(median,thresh1,5,cv2.INPAINT_TELEA)
    plt.imshow(dst)
    plt.show()
    cv2.imwrite('dst.jpg',dst)
    
