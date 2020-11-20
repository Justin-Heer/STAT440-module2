#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 14:24:26 2020

@author: nathaniasantoso
"""

# use automatically configured the ridge regression algorithm
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import csv
import numpy as np
import pandas as pd
from pylab import *
from patsy import dmatrices
# load the dataset
key = 'Z13'
xval = pd.read_csv("/Users/nathaniasantoso/Desktop/Xval-processed.txt", index_col='Id')
xtrain = pd.read_csv("/Users/nathaniasantoso/Desktop/Xtrain-processed.txt", index_col='Id')

yval = pd.read_csv("/Users/nathaniasantoso/Desktop/Yval-processed.txt", index_col='Id').loc[:,key]
ytrain = pd.read_csv("/Users/nathaniasantoso/Desktop/Ytrain-processed.txt").loc[:, key]

# Standarize features
scaler = StandardScaler()
xtrain_std = scaler.fit_transform(xtrain)

# Create ridge regression with three possible alpha values
regr_cv = RidgeCV(alphas=[0.1, 1.0, 10.0])

# Fit the linear regression
model_cv = regr_cv.fit(xtrain_std, ytrain)

# View alpha
model_cv.alpha_
# define model evaluation method
#cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# define model
#model = RidgeCV(alphas=arange(0, 50, 0.01), cv=cv, scoring='neg_mean_absolute_error')
# fit model
#model.fit(xval, yval)
# summarize chosen configuration
#print('alpha: %f' % model.alpha_)

# Select columns
#corrs = xtrain.corrwith(ytrain)
#feature_cols = corrs.sort_values(ascending=False)[0:14].index

#xtrain = xtrain.loc[:, feature_cols]

