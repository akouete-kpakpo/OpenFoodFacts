#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Here we prepare the train and test set 
"""

import numpy as np
import os

from sklearn.model_selection import train_test_split

from logodetection.read import loadX, loadY
from logodetection.config import np_directory


def load_train_set():
    return (np.load(os.path.join(np_directory,'train.npy')),
            np.load(os.path.join(np_directory,'train_labels.npy')))
    
def load_test_set():
    return (np.load(os.path.join(np_directory,'test.npy')),
            np.load(os.path.join(np_directory,'test_labels.npy')))
    
    
if __name__ == '__main__':
    ### LOAD RAW DATA ###
    seed = np.random.seed(7)
    Y_raw = loadY()
    X_raw = loadX()
    
    info = "X_raw, Y_raw shape are {Xshape},{Yshape}"
    print(info.format(Xshape = X_raw.shape, Yshape = Y_raw.shape))
    
    ### GENERATE AND SAVE TRAIN AND TEST SET ####
    X_train, X_test, y_train, y_test = train_test_split(X_raw, Y_raw, 
                                                        test_size=0.30, 
                                                        random_state=seed)
    
    np.save(os.path.join(np_directory,'train.npy'), X_train)
    np.save(os.path.join(np_directory,'test.npy'), X_test)
    np.save(os.path.join(np_directory,'train_labels.npy'), y_train)
    np.save(os.path.join(np_directory,'test_labels.npy'), y_test)
