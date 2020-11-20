# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 14:15:28 2020

@author: justi
Tunes the model for Z09 response variable
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from matplotlib import pyplot as plt

# Import data
KEY = "Z09"

Xtrain = pd.read_csv('processed-data\\Xtrain-processed.txt', index_col='Id')
Xval = pd.read_csv('processed-data\\Xval-processed.txt', index_col='Id')
Xtest = pd.read_csv('processed-data\\Xtest-processed.txt', index_col='Id')

Ytrain = pd.read_csv('processed-data\\Ytrain-processed.txt',
                     index_col='Id').loc[:, KEY]
Yval = pd.read_csv('processed-data\\Yval-processed.txt').loc[:, KEY]

# Select columns
corrs = Xtrain.corrwith(Ytrain)
feature_cols = corrs.sort_values(ascending=False)[0:50].index

Xtrain = Xtrain.loc[:, feature_cols]
Xval = Xval.loc[:, feature_cols]
# Model tuning

params = np.linspace(0,20,30)
maes = []

dtrain = xgb.DMatrix(Xtrain, label=Ytrain)
dval = xgb.DMatrix(Xval, label=Yval)

for param in params:
    model_params = {'objective': 'reg:squarederror',
                    'gamma': 8.5,
                    'max_depth': 200,
                    'seed':1,
                    'min_child_weight': 5,
                    'lambda': param
                    }

    model_params['nthread'] = 3
    model_params['eval_metric'] = 'mae'
    evallist = [(dtrain, 'train'), (dval, 'eval')]
    
    num_rounds = 20
    print("\nparam = {}".format(param))
    model = xgb.train(model_params, dtrain, num_rounds, evallist,
                      early_stopping_rounds=5, verbose_eval=False)


    print("Model fit")

    Ypred = model.predict(dval, ntree_limit=model.best_ntree_limit)
    maes.append(mean_absolute_error(Yval, Ypred))

plt.plot(params, maes)


