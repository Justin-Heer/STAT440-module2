# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 21:32:08 2020

@author: Justin

"""

# Imports

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score

# Import data

Xtrain = pd.read_csv('processed-data\\Xtrain-processed.txt', index_col='Id')
Xval = pd.read_csv('processed-data\\Xval-processed.txt', index_col='Id')
Xtest = pd.read_csv('processed-data\\Xtest-processed.txt', index_col='Id')

Ytrain = pd.read_csv('processed-data\\Ytrain-processed.txt',
                     index_col='Id').loc[:, 'Z02']
Yval = pd.read_csv('processed-data\\Yval-processed.txt').loc[:, 'Z02']

# Select columns
corrs = Xtrain.corrwith(Ytrain)
feature_cols = corrs.sort_values(ascending=False)[0:14].index

Xtrain = Xtrain.loc[:, feature_cols]
Xval = Xval.loc[:, feature_cols]
# Model fit

rfc = RandomForestClassifier(n_estimators=40, random_state=1,
                             verbose=1, n_jobs=2)

rfc.fit(Xtrain,Ytrain)

Ypred = rfc.predict(Xval)

print(classification_report(Yval,Ypred))

