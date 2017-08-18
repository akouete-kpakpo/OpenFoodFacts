#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This class reads the data stored on disk
"""

from os.path import join, exists
from os import listdir
import numpy as np
import pandas as pd
from itertools import chain
from PIL import Image
from keras.preprocessing.image import img_to_array

from logodetection.config import image_directory, np_directory

def read_from_disk(path = image_directory, npixel = (96, 96), 
                   nimages = 1044, nbatches = 10):
    """
        Reads all the images data stored on the disk into one array
        
        Arguments:
            *path: the path of the disk directory containing the images data
            *npixel: a tuple (image_width, image_height) in number of pixels
            *nimages
            *nbatches
        
        Returns:
            *data: a numpy array where each example is one image 
    """

    assert exists(path)
    print("Attention! Chargement de {} images! \n".format(2*nimages))
    print("Cette opération peut prendre du temps.")
    
    images = pd.Series(listdir(path))
    
    # This part is to have one label after the other
    b = list(map(lambda x: x.startswith('a'), images))
    auchan_images = images[b]
    carrefour_images = images[[not x for x in b]]
    images = pd.Series(list(chain.from_iterable(zip(auchan_images.iloc[:nimages], 
                                                    carrefour_images.iloc[:nimages]))))
    
    nsubimg = 3*4
    def _add_img_to_array(imgs,img_name):
        info = "Ajout de {nsubimg} sous-images de l'image {img_name} au dataset"
        print(info.format(nsubimg = nsubimg,img_name = img_name))
        img = Image.open(join(path, img_name))
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

def read_np_data_from_disk(path):
    """
        Reads a numpy file stored on disk
        
        Arguments:
            *path: the path of the numpy data on the disk
        
        Returns:
            *data: as a numpy array
    """
    assert exists(path)
    data = np.load(path)
    return data

if __name__ == '__main__':
    
    imgs_np_path = join(np_directory, 'images.npy')
    labels_np_path = join(np_directory, 'labels.npy')
    
    #### GENERATE DATA AND SAVE ####
    (data, labels) = read_from_disk()
    #TODO: cette opération prend du temps
    #Afficher une jauge de progression plutôt que le nom des images ajoutées
    np.save(imgs_np_path, data)
    np.save(labels_np_path, labels)
    
    ##### LOAD DATA #####
    Y = read_np_data_from_disk(labels_np_path)
    X = read_np_data_from_disk(imgs_np_path)