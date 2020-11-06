# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 20:29:17 2020

@author: Justin

Tunes the model for Z01 response variable
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from matplotlib import pyplot as plt

# Import data

Xtrain = pd.read_csv('processed-data\\Xtrain-processed.txt', index_col='Id')
Xval = pd.read_csv('processed-data\\Xval-processed.txt', index_col='Id')
Xtest = pd.read_csv('processed-data\\Xtest-processed.txt', index_col='Id')

Ytrain = pd.read_csv('processed-data\\Ytrain-processed.txt',
                     index_col='Id').loc[:, 'Z01']
Yval = pd.read_csv('processed-data\\Yval-processed.txt').loc[:, 'Z01']

# Select columns
corrs = Xtrain.corrwith(Ytrain)
feature_cols = corrs.sort_values(ascending=False)[0:14].index

Xtrain = Xtrain.loc[:, feature_cols]

# Model tuning

params = np.arange(5, 100, 10)
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
