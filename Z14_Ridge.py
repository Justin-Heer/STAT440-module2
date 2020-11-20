#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 13:18:47 2020

@author: nathaniasantoso
"""

#load required libraries
import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt
#import sklearn.impute

## Making a list of missing value types
#missing_values = ["n/a", "na", "--", "?"]
##import data frame
#xtest = pd.read_csv("/Users/nathaniasantoso/Downloads/module2/Xtest.txt", sep = " ", na_values = missing_values)
#xtrain = pd.read_csv("/Users/nathaniasantoso/Downloads/module2/Xtrain.txt", sep =" ", na_values = missing_values)

#ytest = pd.read_csv("/Users/nathaniasantoso/Downloads/module2/Ytest.txt", sep =" ", na_values = missing_values)
#ytrain = pd.read_csv("/Users/nathaniasantoso/Downloads/module2/Ytrain.txt", sep =" ", na_values = missing_values)

##use kkn to impute missing value based on nearest neighbours
#from sklearn.impute import KNNImputer
#nan = np.nan
#imputer = KNNImputer(n_neighbors=2, weights="uniform")
#xtest_filled = imputer.fit_transform(xtest)


key = 'Z14'

xval = pd.read_csv("/Users/nathaniasantoso/Desktop/Xval-processed.txt", index_col='Id')
xtrain = pd.read_csv("/Users/nathaniasantoso/Desktop/Xtrain-processed.txt", index_col='Id')

yval = pd.read_csv("/Users/nathaniasantoso/Desktop/Yval-processed.txt", index_col='Id').loc[:,key]
ytrain = pd.read_csv("/Users/nathaniasantoso/Desktop/Ytrain-processed.txt").loc[:, key]





# Select columns
corrs = xtrain.corrwith(ytrain)
feature_cols = corrs.sort_values(ascending=False)[0:14].index

xtrain = xtrain.loc[:, feature_cols]
xval = xval.loc[:, feature_cols]


# Model fit
#ridge regression
rr = Ridge(alpha=0.0001)
rr.fit(xtrain, ytrain) 
pred_train_rr= rr.predict(xtrain)
print(np.sqrt(mean_squared_error(ytrain,pred_train_rr)))
print(r2_score(ytrain, pred_train_rr))

pred_test_rr= rr.predict(xval.loc[:, feature_cols])
print(np.sqrt(mean_squared_error(yval,pred_test_rr))) 
print(r2_score(yval, pred_test_rr))


xtest = pd.read_csv("/Users/nathaniasantoso/Desktop/Xtest-processed.txt", index_col='Id')
xtest = xtest.loc[:, feature_cols]

testPred = pd.DataFrame(rr.predict(xtest), index=xtest.index,columns=['value'])
testPred.to_csv("/Users/nathaniasantoso/Desktop/"+key+".txt")

# calculate mean square error 
mean_squared_error_ridge = np.mean((testPred - xtest)**2) 
print(mean_squared_error_ridge) 
