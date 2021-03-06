# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 20:29:17 2020

@author: Justin

Tunes the model for Z01 response variable
"""

import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
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

params = np.linspace(0, 0.99999, 20)
maes = []
for param in params:
    print("param = {}".format(param))
    print("Model Initialization \n")
    model = MLPRegressor(hidden_layer_sizes=5, alpha=0.4,
                         learning_rate='invscaling', random_state=2,
                         beta_1 = param)

    print("Model fit")
    model.fit(Xtrain, Ytrain)

    Ypred = model.predict(Xval.loc[:, feature_cols])

    maes.append(mean_absolute_error(Yval, Ypred))

plt.plot(params, maes)
