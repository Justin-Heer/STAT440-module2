# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 18:03:27 2020

@author: Justin

Z03 model training script
"""
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


# Import data
key = 'Z03'
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

model = RandomForestRegressor(n_estimators=50, verbose=1, n_jobs=4)

model.fit(Xtrain, Ytrain)

Ypred = model.predict(Xval.loc[:, feature_cols])

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
