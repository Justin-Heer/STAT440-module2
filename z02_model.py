# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 20:29:26 2020

@author: Justin
z02 response variable model, this variable is a binary classification variable

"""


# Imports

import pandas as pd

from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Import data
key = 'Z02'
print("Importing Data")

Xtrain = pd.read_csv('processed-data\\Xtrain-processed.txt', index_col='Id')
Xval = pd.read_csv('processed-data\\Xval-processed.txt', index_col='Id')


Ytrain = pd.read_csv('processed-data\\Ytrain-processed.txt',
                     index_col='Id').loc[:, key]
Yval = pd.read_csv('processed-data\\Yval-processed.txt').loc[:, key]


# Select columns
corrs = Xtrain.corrwith(Ytrain)
feature_cols = corrs.sort_values(ascending=False)[0:14].index

Xtrain = Xtrain.loc[:, feature_cols]
Xval = Xval.loc[:, feature_cols]
# Model fit
print("Fitting model")

model = SVC(kernel='sigmoid')

model.fit(Xtrain, Ytrain)

Ypred = model.predict(Xval)

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
    Xtest = pd.read_csv('processed-data\\Xtest-processed.txt', index_col='Id')
    Xtest = Xtest.loc[:, feature_cols]

    testPred = pd.DataFrame(model.predict(Xtest), index=Xtest.index,
                            columns=['value'])
    testPred.to_csv("recent-predictions\\"+key+".txt")

print(' done')