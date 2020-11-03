# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 20:29:17 2020

@author: Justin

Trains the model for Z01 response variable
"""

import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

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

# Model training

print("Model Initialization \n")
rfr = RandomForestRegressor(n_estimators=10, verbose=1)

print("Model fit")
rfr.fit(Xtrain, Ytrain)

Ypred = rfr.predict(Xval.loc[:, feature_cols])

mean_absolute_error(Yval, Ypred)

