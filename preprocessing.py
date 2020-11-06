# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 20:33:01 2020

@author: Justin

Preprocessing script for models. 
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split


class CustomImputer(BaseEstimator, TransformerMixin):
    '''
    Takes irregular missing values and replaces with np.nan
    '''
    def __init__(self, missing_values = None):
        self.missing_values = missing_values 

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        print('\n transform() called in ', self)
        for missing_value in self.missing_values:
            X.replace(missing_value, np.nan, inplace=True)

        return X


class z02Imputer(BaseEstimator, TransformerMixin):
    '''
    Takes erroneous z02 values 7 and replaces with 1
    '''

    def __init__(self):
        None

    def fit(self, y):
        return self

    def transform(self, y):
        print("\n transform called in, ",self)
        inx = y.loc[:,'Z02'] == 7
        y.loc[inx,'Z02'] = 1

        return y
        

# Import data


Xtrain = pd.read_csv('original-data\\Xtrain.txt', sep=' ', index_col='Id',
                     low_memory=False)

Xtest = pd.read_csv('original-data\\Xtest.txt', sep=' ', index_col='Id',
                    low_memory=False)

Ytrain = pd.read_csv('original-data\\Ytrain.txt', delimiter=' ',
                     index_col='Id')

Ytest = pd.read_csv('original-data\\Ytest.txt')


Xtrain, Xval, Ytrain, Yval = train_test_split(Xtrain, Ytrain, random_state=1,
                                              test_size=0.2)

X_pipeline_steps = [('custom imputer', CustomImputer(missing_values=['?'])),
                    ('simple imputer', SimpleImputer(copy=False)), ]

X_pipeline = Pipeline(steps = X_pipeline_steps)
Xtrain[:] = X_pipeline.fit_transform(Xtrain)
Xval[:] = X_pipeline.transform(Xval)

Y_pipeline_steps = [('z02Imputer', z02Imputer()),
                    ('simple imputer', SimpleImputer(copy=False)),
                    ]
Y_pipeline = Pipeline(steps=Y_pipeline_steps)
Ytrain[:] = Y_pipeline.fit_transform(Ytrain)
Yval[:] = Y_pipeline.transform(Yval)

Xtest[:] = X_pipeline.transform(Xtest)


Xtrain.to_csv('processed-data\\Xtrain-processed.txt')
Xval.to_csv('processed-data\\Xval-processed.txt')
Ytrain.to_csv('processed-data\\Ytrain-processed.txt')
Yval.to_csv('processed-data\\Yval-processed.txt')
Xtest.to_csv('processed-data\\Xtest-processed.txt')


