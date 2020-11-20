# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 20:24:51 2020

@author: Justin

This is the script that combines the individual model outputs scripts
for response variables Z01,Z02,Z03 and Z04 ... should work for the rest as
well but is untested ?

"""
# Imports
import glob
from pandas import read_csv

# GET SAVE LOCATION
SAVE_PATH = 'best-predictions\\kaggle-submission.txt'
# Get paths
zPaths = glob.glob('recent-predictions\\*.txt')

# Sort paths by Z num
zPaths.sort(key=lambda x: x[-4:-6])

# read in predictions
dfs = [read_csv(zPath, index_col='Id') for zPath in zPaths]

pred_df = dfs[0]

for inx, df in enumerate(dfs[1:]):
    pred_df[str(inx+2)] = df['value']

columns = ["Z01", "Z02", 'Z03', 'Z04', 'Z05', 'Z06', 'Z07', 'Z08', 'Z09',
           'Z10', 'Z11', 'Z12', 'Z13', 'Z14']

pred_df.columns = columns

# Import Ytest-processed

Ytest = read_csv("processed-data\\Ytest-processed.txt", index_col='Id')

for index in Ytest.index:
    r_col = Ytest.loc[index, 'r_col']

    Ytest.loc[index, 'Value'] = pred_df.loc[index, r_col]

Ytest['Value'].to_csv(SAVE_PATH)
