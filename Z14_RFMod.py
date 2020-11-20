#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 13:48:52 2020

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
Xval = Xval.loc[:, feature_cols]
# Model fit
print("Fitting model")

model = RandomForestRegressor(n_estimators=170, verbose=1, n_jobs=4)

model.fit(Xtrain, Ytrain)

Ypred = model.predict(Xval.loc[:, feature_cols])

# Ask if test predictions need to be generated
while True:
    ans = input("\n Do you want to produce test set predictions (y/n)?   => ")
    try:
        ans = str(ans)
    except ValueError:
        print("\n Not a letter")
        continue
    if ans == 'y' or ans == 'n':
        break
    else:
        print('\n enter (y/n)')


if ans == 'y':
    Xtest = pd.read_csv("/Users/nathaniasantoso/Desktop/Xtest-processed.txt", index_col='Id')
    Xtest = Xtest.loc[:, feature_cols]

    testPred = pd.DataFrame(model.predict(Xtest), index=Xtest.index,
                            columns=['value'])
    testPred.to_csv("/Users/nathaniasantoso/Desktop/"+key+"_RF"+".txt")

print(' done')