#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Use deep learning models for logo detection
"""

#import numpy as np
#import os

from keras.models import Sequential
from keras.layers import Dense

from logodetection.preprocessing import load_train_set, load_test_set

(X_train, y_train) = load_train_set()
info = "X_train, y_train shape are {Xshape},{Yshape}"
print(info.format(Xshape = X_train.shape, Yshape = y_train.shape))

#Y = Y.ravel()
#Y.shape = (24960,1)
#b = Y==1
#a = np.concatenate((b,~b), axis = 1)
#Y = a

### MODEL ####
model = Sequential()
model.add(Dense(12, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, validation_split=0.30, epochs=15, batch_size=10)

# Evaluate the model
(X_test, y_test) = load_test_set()
scores = model.evaluate(X_test, y_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))