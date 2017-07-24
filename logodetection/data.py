# -*- coding: utf-8 -*-

""" Commence le tra vail sur les données de OpenFoodFacts
    
    1- Se constituer un jeu d'apprentissage labellisé (télécharger 1000 - 
       10 000 images de logos: moitié Carrefour, moitié Auchan)
    
    2- Entraîner réseau neuronal (je suis le professeur)
    
    3- Test
    
    4- Utiliser la  librairie Python "keras": code en 10 lignes

"""
import numpy as np
import pandas as pd
import requests
from PIL import Image
from io import BytesIO, StringIO
import os

###########################
##Fonction auxiliaire ####
##########################

def image_url_to_nparray(image_url, image_name = 'image', npixel=(30,40),
     save = False):
    """
    Downloads the image from image_url and saves it
    in the working directory
    
    Arguments:
    * image_url: the url of the image
    * npixel: a tuple (image_width, image_height) in number of pixels
    * image_name: the name to save in the wd for the downloaded image
    * save: a boolean. If True, we save the image
    
    Returns:
    * data: the image in numpy array format
    
    """
    r = requests.get(image_url)
    pic = Image.open(BytesIO(r.content))
    if save:
        path = '/home/kevin/Desktop/OpenFoodFacts/OpenFoodFacts/data/images'
        pic.save(os.path.join(path, image_name))
        print("Saving the image {} in working directory \n".format(image_name))
    
#    pic = pic.resize((npixel[0],npixel[1]))
#    data_as_array = np.asarray(pic)
    data_as_array = np.expand_dims(pic, axis=0)
    
    if data_as_array.shape != (1, npixel[1],npixel[0],3):
        #If problem of shape I assume the image is totally black
        data = np.zeros((1,npixel[1],npixel[0],3))
    else:
        data = data_as_array
        
    return data


#####################
##### DATA  ########
####################
read_or_generate_data = False
save_image = True
npixel = (300,400)
wd_path = '/home/kevin/Desktop/OpenFoodFacts/OpenFoodFacts/data/csv'

if read_or_generate_data:
    data_path = os.path.join(wd_path, 'data.csv')
    assert os.path.exists(data_path)
    mydata = pd.read_csv(data_path)
    Y = np.array(mydata['label'])
    X = np.array(mydata.drop('label'))

else:  
    url = "http://world.openfoodfacts.org/data/en.openfoodfacts.org.products.csv"
    s = requests.get(url).content
    whole_data = pd.read_csv(StringIO(s.decode('utf-8')), sep='\t')
    
    #Keep two brands only: Carrefour = 1 and Auchan = 0
    two_classes_data = whole_data.query("brands == 'Carrefour'|brands == 'Auchan'")
    two_classes_data = two_classes_data.reset_index()
    two_classes_data = two_classes_data.loc[:,['brands','image_url',
                                               #'image_small_url'
                                              ]
                                           ]
    
    #two_classes_data.brands.value_counts()
    two_classes_data.brands = two_classes_data.brands.str.replace('Carrefour', '1')
    two_classes_data.brands = two_classes_data.brands.str.replace('Auchan', '0')
    
    two_classes_data = two_classes_data.loc[two_classes_data.image_url.notnull(),:]
    two_classes_data = two_classes_data.reset_index()
    del two_classes_data['index']
    

    npixel = (300,400)
    wd_path = '/home/kevin/Desktop/OpenFoodFacts/OpenFoodFacts/data/csv'
    save_image = True
 
    imgs = np.empty(shape = (0, npixel[1], npixel[0], 3), dtype = np.uint8)
    labels = np.empty(shape = (0,), dtype = np.bool)
    
    data = two_classes_data.head(10)
    print("On va télécharger {} images. \n".format(len(two_classes_data.image_url)) )
    print("Attention, cette opération peut prendre du temps")
   
    for row in data.itertuples():
        if row.brands == '1':
            image_name = 'carrefour{}.jpg'.format(str(row.Index))
        else:
            image_name = 'auchan{}.jpg'.format(str(row.Index))
        image_url = row.image_url
        x = image_url_to_nparray(image_url,image_name)
        print(row)
        imgs = np.vstack([imgs, x])
        labels = np.concatenate((labels, [row.brands]))
    
    np_path = '/home/kevin/Desktop/OpenFoodFacts/OpenFoodFacts/data/npy'
    imgs_np_path = os.path.join(np_path, 'images.npy')
    labels_np_path = os.path.join(np_path, 'labels.npy')
    np.save(imgs_np_path, imgs)
    np.save(labels_np_path, labels)


