# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 18:18:39 2020

@author: Justin

Model training Z04


"""
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


# Import data
key = 'Z04'
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

model = RandomForestRegressor(random_state=1,
                              n_estimators=70,
                              max_depth=20,
                              min_samples_split=0.0003)

model.fit(Xtrain, Ytrain)

Ypred = model.predict(Xval.loc[:, feature_cols])
print("MAE=  {}".format(mean_absolute_error(Yval, Ypred)))

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
    Xtrain = pd.read_csv('processed-data\\Xtrain-full-processed.txt',
                         index_col='Id')
    Xtrain = Xtrain.loc[:, feature_cols]
    Ytrain = pd.read_csv('processed-data\\Ytrain-full-processed.txt',
                         index_col='Id').loc[:, key]
    print("Fitting model")
    model.fit(Xtrain, Ytrain)
    Xtest = pd.read_csv('processed-data\\Xtest-processed.txt', index_col='Id')
    Xtest = Xtest.loc[:, feature_cols]

    testPred = pd.DataFrame(model.predict(Xtest), index=Xtest.index,
                            columns=['value'])
    testPred.to_csv("recent-predictions\\"+key+".txt")

print(' done')
