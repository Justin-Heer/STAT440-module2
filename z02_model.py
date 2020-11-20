# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 18:18:39 2020

@author: Justin

Model training Z09


"""
import pandas as pd

import xgboost as xgb
from sklearn.metrics import accuracy_score


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
feature_cols = corrs.sort_values(ascending=False)[0:50].index

Xtrain = Xtrain.loc[:, feature_cols]
Xval = Xval.loc[:, feature_cols]

dtrain = xgb.DMatrix(Xtrain, label=Ytrain)
dval = xgb.DMatrix(Xval, label=Yval)

# Model fit
print("Fitting model")

model_params = {'objective': 'binary:hinge',
                'lambda': 100,
                'max_depth': 1200,
                'min_child_weight': 55,
                'eta': 0.2,
                'gamma':10
                }


model_params['nthread'] = 4
model_params['eval_metric'] = 'auc'
evallist = [(dtrain, 'train'), (dval, 'eval')]

num_rounds = 100

model = xgb.train(model_params, dtrain, num_rounds, evallist,
                  early_stopping_rounds=30, verbose_eval=True)


num_rounds = model.best_ntree_limit
Ypred = model.predict(dval, ntree_limit=model.best_ntree_limit)

print("Acc_score=  {}".format(accuracy_score(Yval, Ypred)))

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
