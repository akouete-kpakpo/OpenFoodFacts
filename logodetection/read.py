#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 11:44:25 2017

@author: kevin
"""

import os, os.path
import numpy as np
import pandas as pd
from itertools import chain
from PIL import Image
from keras.preprocessing.image import (img_to_array, 
                                       #load_img
                                       )

def read_from_disk(path, npixel = (96, 96), nimages = 1044, nbatches = 10):
    """
        Reads the images data stored on the disk
        
        Arguments:
            *path: the path of the disk directory containing the images data
            *npixel: a tuple (image_width, image_height) in number of pixels
            *nimages
            *nbatches
        
        Returns:
            *data: a numpy array where each example is one image 
    """

    assert os.path.exists(path)
    print("Attention! Chargement de {} images! \n".format(2*nimages))
    print("Cette opération peut prendre du temps.")
    
    images = pd.Series(os.listdir(path))
    b = list(map(lambda x: x.startswith('a'), images))
    auchan_images = images[b]
    carrefour_images = images[[not x for x in b]]
    images = pd.Series(list(chain.from_iterable(zip( auchan_images.iloc[:nimages], 
                                          carrefour_images.iloc[:nimages]))))
    def _add_img_to_array(imgs,img_name):
        print("Ajout de l'image {} au dataset".format(img_name))
        img = Image.open(os.path.join(path,img_name))
        img = img.resize((npixel[0]*3,npixel[1]*4))
        
        if img_to_array(img).shape != (npixel[1]*4,npixel[0]*3,3):
            #The image is black
            print("The image {} is black.".format(img_name))
            imgs = np.vstack([imgs, np.zeros((12,npixel[1],npixel[0],3))])
        else:
            width = npixel[0]
            height = npixel[1]
            #imgs = np.empty(shape = (0,npixel[1],npixel[0],3), dtype = np.uint8)
            for nleft in range(3):
                left = nleft*96
                for ntop in range(4):
                    top = ntop*96
                    box = (left, top, left+width, top+height)
                    subimg = img.crop(box)
                    #print("Ajout de la portion  {} de cette image au dataset".format((nleft, ntop)))
                    x = img_to_array(subimg)
                    x = x.astype(np.uint8)
                    #x = x.reshape(1,-1)
                    x = np.expand_dims(x, axis=0)
                    imgs = np.vstack([imgs, x])
        
        return imgs
    
    nsubimg = 3*4
    n_images_per_batch = 2*nimages//nbatches
    n_sub_images_per_batch = 2*nimages//nbatches*nsubimg
    data = np.empty(shape = (0,n_sub_images_per_batch,npixel[1],npixel[0],3))
    labels = np.empty(shape = (0,n_sub_images_per_batch))
    for batch in range(nbatches):
        s = np.arange(n_images_per_batch*batch,n_images_per_batch*(batch+1))
        imgs = np.empty(shape = (0, npixel[1],npixel[0],3), dtype = np.uint8)
        batch_label = np.array([], dtype = np.bool_)
        for f in images[s]:
            if f.startswith('a'):
                #Auchan coded with 0
                batch_label = np.hstack([batch_label, np.zeros(nsubimg, dtype = np.bool_)])
            else:
                batch_label = np.hstack([batch_label, np.ones(nsubimg, dtype = np.bool_)])
            imgs = _add_img_to_array(imgs,f)
        imgs = np.expand_dims(imgs, axis=0)
        batch_label = np.expand_dims(batch_label, axis=0)
        data = np.vstack([data, imgs])
        labels = np.vstack([labels, batch_label])
    
    return data, labels

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
    images_path = "/home/kevin/Desktop/OpenFoodFacts/data/images"
    csv_path = "/home/kevin/Desktop/OpenFoodFacts/data/csv"
    np_path = '/home/kevin/Desktop/OpenFoodFacts/data/npy'
    imgs_np_path = os.path.join(np_path, 'images.npy')
    labels_np_path = os.path.join(np_path, 'labels.npy')
    npixel = (96,96)
    (data, labels) = read_from_disk(images_path)
    #TODO: cette opération prend du temps
    #Afficher une jauge de progression plutôt que le nom des images ajoutées
    np.save(imgs_np_path, data)
    np.save(labels_np_path, labels)
    #data2 = read_csv_data_from_disk(os.path.join(csv_path, 'data.csv'))
    Y = np.load(labels_np_path)
    X = np.load(imgs_np_path)