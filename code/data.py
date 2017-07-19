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
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

#####################
##### DATA  ########
####################

read_or_generate_data = True
save_image = False
npixel = (96,96)
wd_path = '/home/kevin/Desktop/OpenFoodFacts/OpenFoodFacts/data/csv'

if read_or_generate_data:
    data_path = os.path.join(wd_path, 'data.csv')
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
    two_classes_data = two_classes_data.loc[:,['brands','image_url','image_small_url']]
    #two_classes_data.brands.value_counts()
    two_classes_data.brands = two_classes_data.brands.str.replace('Carrefour', '1')
    two_classes_data.brands = two_classes_data.brands.str.replace('Auchan', '0')
    
    two_classes_data = two_classes_data.loc[two_classes_data.image_url.notnull(),:]
    two_classes_data = two_classes_data.reset_index()
    del two_classes_data['index']
    
    #TODO: pour les images de mauvaises tailles regarder en détail
    #image_url = 'http://en.openfoodfacts.org/images/products/324/541/417/2081/front.4.400.jpg'
    def image_url_to_nparray(image_url, npixel=npixel, image_name = 'image', 
                             save = save_image):
        """
            Downloads the image from image_url and saves it
            in the working directory
            
            Arguments:
                * image_url: the url of the image
                * npixel: a tuple (image_width, image_height) in number of pixelq
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
            print("Saving the image {} in working directory".format(image_name))
            
        pic = pic.resize((npixel[0],npixel[1]))
        data_as_array = np.asarray(pic)
        
        if data_as_array.shape == (npixel[0],npixel[1],3):
            data = data_as_array.reshape(1, npixel[0]*npixel[1]*3)
        else:
            #If problem of shape I assume the image is totally black
            data = np.zeros(npixel[0]*npixel[1]*3)
        
        
        return data
    
    print("On va télécharger {} images. \n".format(len(two_classes_data.image_url)) )
    print("Attention, cette opération peut prendre du temps")
    mydata = two_classes_data.apply(lambda x: image_url_to_nparray(x['image_url']), x['image_url'][44:].replace('/',''), axis = 1)
    
    trainingY = two_classes_data.brands
    
    X = np.array([mydata[_].ravel() for _ in range(mydata.shape[0])])
    Y = np.array(trainingY, dtype = 'uint16')
    
    X_to_pandas = pd.DataFrame(X)
    Y_to_pandas = pd.DataFrame(Y)
    X_to_pandas['label'] = Y_to_pandas
    X_to_pandas.to_csv(os.path.join(wd_path, 'data.csv'), 
                       index = False, encoding = 'utf8')



##########################
#### DEEP LEARNING   ####
##########################

# fix random seed for reproducibility
np.random.seed(7)

# create model
model = Sequential()
model.add(Dense(12, input_dim=X.shape[1], activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
## Manual split
#X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=seed)
#model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=150, batch_size=10)
# Auto split
model.fit(X, Y, validation_split=0.30, epochs=150, batch_size=10)

# evaluate the model
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))