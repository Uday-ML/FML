from Regression import Regression
from Utility_functions import TrainTestSplit,Accuracy,MeanSquaredError
import numpy as np
from sklearn.datasets import load_iris,fetch_california_housing
import pandas as pd
from sklearn.metrics import accuracy_score,mean_squared_error
from sklearn.linear_model import LogisticRegression,LinearRegression


import sys





    

    
california_housing_dataset = fetch_california_housing()
X = california_housing_dataset.data
y = california_housing_dataset.target

splitter=TrainTestSplit(test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = splitter.split(X, y)

custom_regressor = Regression(num_variables=X.shape[1],learning_rate=0.005,num_epochs=3000)
custom_regressor.fit(X_train, y_train)

custom_predictions = custom_regressor.predict(X_test)
mse=MeanSquaredError()
custom_mse=mse.calculate(y_test,custom_predictions)
print("Custom Regression Mean Squared Error:", custom_mse)

sklearn_regressor = LinearRegression()
sklearn_regressor.fit(X_train, y_train)

sklearn_predictions = sklearn_regressor.predict(X_test)
sklearn_mse=mse.calculate(y_test,sklearn_predictions)

print("Scikit-Learn Linear Regression Mean Squared Error:", sklearn_mse)
    

    

