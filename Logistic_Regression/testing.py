from LogisticRegression import LR
from Utility_functions import TrainTestSplit,Accuracy,MeanSquaredError
import numpy as np
from sklearn.datasets import load_iris,fetch_california_housing
import pandas as pd
from sklearn.metrics import accuracy_score,mean_squared_error
from sklearn.linear_model import LogisticRegression,LinearRegression






    

    
iris_dataset = load_iris()
X = iris_dataset.data
y = (iris_dataset.target == 2).astype(int)


splitter=TrainTestSplit(test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = splitter.split(X, y)

logistic_regressor = LR(num_variables=X.shape[1], learning_rate=0.1, num_epochs=1000)
logistic_regressor.fit(X_train, y_train)

sklearn_logistic_regressor = LogisticRegression()
sklearn_logistic_regressor.fit(X_train, y_train)
y_pred_sklearn = sklearn_logistic_regressor.predict(X_test)

y_pred = logistic_regressor.predict(X_test)

accu=Accuracy()
accuracy_custom = accu.accuracy_score(y_test, y_pred)
accuracy_sklearn = accu.accuracy_score(y_test, y_pred_sklearn)

print(f"Logistic Regression  Accuracy Custom: {accuracy_custom}")
print(f"Logistic Regression Accuracy SKlearn: {accuracy_sklearn}")

