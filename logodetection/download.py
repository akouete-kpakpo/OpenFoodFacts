# -*- coding: utf-8 -*-

""" Commence le travail sur les données de OpenFoodFacts
    
    1- Se constituer un jeu d'apprentissage labellisé (télécharger 1000 - 
       10 000 images de logos: moitié Carrefour, moitié Auchan)
    
    2- Entraîner réseau neuronal (je suis le professeur)
    
    3- Test
    
    4- Utiliser la  librairie Python "keras": code en 10 lignes

"""
import pandas as pd
import requests
from PIL import Image
from io import BytesIO, StringIO
import os

from logodetection.config import dump_csv_url, image_directory

def download_image(url, image_name, destination):
    """
    Downloads the image from url and saves it
    in the working directory
    
    Arguments:
    * url (str) : the url of the image
    * image_name (str) : the name of the downloaded image in the destination directory
    * destination (str) : the path of the destination directory
    
    Returns:
    * None
    
    """
    r = requests.get(url)
    pic = Image.open(BytesIO(r.content))
    pic.save(os.path.join(destination, image_name))
    print("Saving the image {} in directory \n".format(image_name))
    return None

def download_all_data(url = dump_csv_url, destination = image_directory):
    """
    Downloads all images available on the dump of OpenFoodFacts website
    
    Arguments:
        * url : the csv dump url
        * destination (str) : the path of directory where to store all the 
        images
    """
    
    def download_csv(url):
        """
        Downloads the csv on the dump of OpenFoodFacts.
        Returns the corresponding pandas dataframe.
        """
        s = requests.get(url).content
        print("Downloading 1GB file. It may take time depending your" +
              "internet connexion")
        return pd.read_csv(StringIO(s.decode('utf-8')), sep='\t')
        
    def keep_two_labels(df):
        """
        Keeps the two most appearing labels
        """
        assert 'brands' in df
        
        df['brands'].value_counts()
        # Carrefour 3525
        # Auchan 2987
        # Keep the two top brands 
        two_labels_data = df.query("brands == 'Carrefour'|brands == 'Auchan'")
        two_labels_data = two_labels_data.reset_index()
        
        # Encode the kept labels with boolean: Carrefour = 1 and Auchan = 0
        two_labels_data.brands = two_labels_data.brands.str.replace('Carrefour', '1')
        two_labels_data.brands = two_labels_data.brands.str.replace('Auchan', '0')
        
        # Keep the lines of the df where we can find a image url
        two_labels_data = two_labels_data.loc[two_labels_data.image_url.notnull(),:]
        two_labels_data = two_labels_data.reset_index()
        
        return two_labels_data
        
    def keep_features(df, features = ['brands','image_url']):
        """
        Keeps the features
        """
        return df.loc[:, features]
    
    whole_data = download_csv(url)
    two_labels_data = keep_two_labels(whole_data)
    data = keep_features(two_labels_data, features = ['brands','image_url'])
    
    print("On va télécharger {} images. \n".format(len(data.image_url)) )
    print("Attention, cette opération peut prendre du temps")
   
    for row in data.itertuples():
        if row.brands == '1':
            image_name = 'carrefour{}.jpg'.format(str(row.Index))
        else:
            image_name = 'auchan{}.jpg'.format(str(row.Index))
        image_url = row.image_url
        
        download_image(image_url, image_name, destination)
        print(row)
        
if __name__ == '__main__':
    download_all_data()