# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 14:15:28 2020

@author: justi
Tunes the model for Z05 response variable
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# Import data
KEY = "Z05"

Xtrain = pd.read_csv('processed-data\\Xtrain-processed.txt', index_col='Id')
Xval = pd.read_csv('processed-data\\Xval-processed.txt', index_col='Id')
Xtest = pd.read_csv('processed-data\\Xtest-processed.txt', index_col='Id')

Ytrain = pd.read_csv('processed-data\\Ytrain-processed.txt',
                     index_col='Id').loc[:, KEY]
Yval = pd.read_csv('processed-data\\Yval-processed.txt').loc[:, KEY]

# Select columns
corrs = Xtrain.corrwith(Ytrain)
feature_cols = corrs.sort_values(ascending=False)[0:14].index

Xtrain = Xtrain.loc[:, feature_cols]
Xval = Xval.loc[:, feature_cols]
# Model tuning

params = np.arange(1,100,10)
maes = []

dtrain = xgb.DMatrix(Xtrain, label=Ytrain)
dval = xgb.DMatrix(Xval, label=Yval)

for param in params:
    print("\nparam = {}".format(param))
    model = RandomForestRegressor(n_estimators = param)
    
    print("Model fit")
    model.fit(Xtrain,Ytrain)
    
    Ypred = model.predict(Xval)
    maes.append(mean_absolute_error(Yval, Ypred))

plt.plot(params, maes)


