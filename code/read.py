#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 11:44:25 2017

@author: kevin
"""

import os, os.path
import numpy as np
from PIL import Image
from keras.preprocessing.image import (img_to_array, 
                                       #load_img
                                       )

def read_from_disk(path, npixel = (300, 400)):
    """
        Reads the images data stored on the disk
        
        Arguments:
            *path: the path of the disk directory containing the images data
            *npixel: a tuple (image_width, image_height) in number of pixels
        
        Returns:
            *data: a numpy array where each example is one image 
    """
    imgs = np.empty(shape = (1, npixel[1], npixel[0], 3) )
    #image_list = map(Image.open, glob('your/path/*.gif'))
    assert os.path.exists(path)
    print("Attention! Chargement de {} images! \n".format(len(os.listdir(path))))
    print("Cette opération peut prendre du temps.")
    
    for f in os.listdir(path):
        print("Ajout de l'image {} au dataset".format(f))
        img = Image.open(os.path.join(path,f))
        img = img.resize((npixel[0],npixel[1]))
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)
        if x.shape != (1, npixel[1], npixel[0], 3):
            #The image is black
            x = np.zeros((1, npixel[1], npixel[0], 3))
            print("The image {} is black.".format(f))
        imgs = np.vstack([imgs, x])
    
    return imgs

def read_csv_data_from_disk(path):
    """
        Reads a csv file stored on disk
        
        Arguments:
            *path: the path of the csv data on the disk
        
        Returns:
            *data: as a numpy array
    """
    assert os.path.exists(path)
    data = np.genfromtxt(path, delimiter=',')
    return data

if __name__ == '__main__':
    images_path = "/home/kevin/Desktop/OpenFoodFacts/OpenFoodFacts/data/images"
    csv_path = "/home/kevin/Desktop/OpenFoodFacts/OpenFoodFacts/data/csv"
    npixel = (300,400)
    data1 = read_from_disk(images_path)
    np.savetxt(os.path.join(csv_path, "data.csv"),
               data1.reshape((data1.shape[0],-1)), 
               delimiter=",")
    data2 = read_csv_data_from_disk(os.path.join(csv_path, 'data.csv'))