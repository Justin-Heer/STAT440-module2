# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 14:15:28 2020

@author: justi
Tunes the model for Z04 response variable
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from matplotlib import pyplot as plt

# Import data
KEY = "Z04"

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

params = np.linspace(0.000001,0.001,10)
maes = []
for param in params:
    print("param = {}".format(param))
    print("Model Initialization \n")
    model = RandomForestRegressor(random_state=1,
                                  n_estimators=70,
                                  max_depth=20,
                                  min_samples_split=0.0003)

    print("Model fit")
    model.fit(Xtrain, Ytrain)

    Ypred = model.predict(Xval)

    maes.append(mean_absolute_error(Yval, Ypred))

plt.plot(params, maes)


