# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 18:18:39 2020

@author: Justin

Model training Z08


"""
import pandas as pd

import xgboost as xgb
from sklearn.metrics import mean_absolute_error


# Import data
key = 'Z08'
print("Importing Data")

Xtrain = pd.read_csv('processed-data\\Xtrain-processed.txt', index_col='Id')
Xval = pd.read_csv('processed-data\\Xval-processed.txt', index_col='Id')


Ytrain = pd.read_csv('processed-data\\Ytrain-processed.txt',
                     index_col='Id').loc[:, key]
Yval = pd.read_csv('processed-data\\Yval-processed.txt').loc[:, key]


# Select columns
corrs = Xtrain.corrwith(Ytrain)
feature_cols = corrs.sort_values(ascending=False)[0:50].index

Xtrain = Xtrain.loc[:, feature_cols]
Xval = Xval.loc[:, feature_cols]

dtrain = xgb.DMatrix(Xtrain, label=Ytrain)
dval = xgb.DMatrix(Xval, label=Yval)

# Model fit
print("Fitting model")

model_params = {'objective': 'reg:squarederror',
                    'gamma': 8.5,
                    'max_depth': 200,
                    'seed':1,
                    'min_child_weight': 3,
                    'lambda': 200
                    
                    }


model_params['nthread'] = 4
model_params['eval_metric'] = 'mae'
evallist = [(dtrain, 'train'), (dval, 'eval')]

num_rounds = 40

model = xgb.train(model_params, dtrain, num_rounds, evallist,
                  early_stopping_rounds=5, verbose_eval=True)


num_rounds = model.best_ntree_limit
Ypred = model.predict(dval, ntree_limit=model.best_ntree_limit)

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
    dtrain = xgb.DMatrix(Xtrain, label=Ytrain)
    print("Fitting model")
    evallist = [(dtrain, 'train')]
    model = xgb.train(model_params, dtrain, num_rounds, evallist,
                      verbose_eval=True)

    Xtest = pd.read_csv('processed-data\\Xtest-processed.txt', index_col='Id')
    Xtest = Xtest.loc[:, feature_cols]

    dtest = xgb.DMatrix(Xtest)

    testPred = pd.DataFrame(model.predict(dtest), index=Xtest.index,
                            columns=['value'])
    testPred.to_csv("recent-predictions\\"+key+".txt")

print(' done')
