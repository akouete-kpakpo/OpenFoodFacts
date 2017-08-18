#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Use deep learning models for logo detection
"""

import numpy as np
import os

from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split