import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

for i in ['Z09', 'Z10', 'Z11', 'Z12']:
	xTrain = pd.read_csv('data/Xtrain.csv', index_col = 'Id')
	xTest = pd.read_csv('data/Xtest.csv', index_col = 'Id')
	yTrain = pd.read_csv('data/Ytrain.txt', sep = ' ',index_col = 'Id')
	print('{:-^20}'.format(""))
	print('Working on ', i)
	xTrain = xTrain.replace('?', np.nan)
	xTrain["B15"] = xTrain["B15"].astype(float)
	xTrain = xTrain.fillna(xTrain.mean())

	xTest = xTest.replace('?', np.nan)
	xTest["B15"] = xTest["B15"].astype(float)
	xTest = xTest.fillna(xTest.mean())

	yTrain = yTrain.fillna(yTrain.mean())

	param_grid = {
				'max_depth': [30, 50, 60, 70],
				'min_samples_split': [6, 8, 10, 12],
				'n_estimators': [80, 90, 100, 110]
				}
	# Create a based model
	rf = RandomForestRegressor()
	grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)
	# Fit the grid search to the data
	grid_search.fit(xTrain, yTrain)
	grid_search.best_params_

	model = RandomForestRegressor(grid_search.best_params_)

	model.fit(xTrain,yTrain)

	predict = model.predict(xTest)
	df_predict = pd.DataFrame(predict,columns = ['value'],index = xTest.index)
	filename = 'predict_data/%s.txt' % i
	df_predict.to_csv(filename ,sep=',',index = True)