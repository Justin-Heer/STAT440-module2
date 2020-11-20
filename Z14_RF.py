#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 13:18:38 2020

@author: nathaniasantoso
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from matplotlib import pyplot as plt

# Import data

# load the dataset
key = 'Z14'
Xval= pd.read_csv("/Users/nathaniasantoso/Desktop/Xval-processed.txt", index_col='Id')
Xtrain= pd.read_csv("/Users/nathaniasantoso/Desktop/Xtrain-processed.txt", index_col='Id')

Yval = pd.read_csv("/Users/nathaniasantoso/Desktop/Yval-processed.txt", index_col='Id').loc[:,key]
Ytrain = pd.read_csv("/Users/nathaniasantoso/Desktop/Ytrain-processed.txt").loc[:, key]

# Select columns
corrs = Xtrain.corrwith(Ytrain)
feature_cols = corrs.sort_values(ascending=False)[0:14].index

Xtrain = Xtrain.loc[:, feature_cols]

# Model tuning

params = np.arange(5, 200, 10)
maes = []
for param in params:
    print("param = {}".format(param))
    print("Model Initialization \n")
    rfr = RandomForestRegressor(n_estimators=param, verbose=0, n_jobs=4)

    print("Model fit")
    rfr.fit(Xtrain, Ytrain)

    Ypred = rfr.predict(Xval.loc[:, feature_cols])

    maes.append(mean_absolute_error(Yval, Ypred))

plt.plot(params, maes)