from Utility_functions import TrainTestSplit,Accuracy,MeanSquaredError
from Bagging import Bagging
import numpy as np
from sklearn.datasets import load_iris,fetch_california_housing
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,mean_squared_error
from sklearn.ensemble import BaggingClassifier
import sys
sys.path.append('KNN')
from KNN import KNN

iris_df = pd.read_csv('Decision_Tree/iris.csv')  # Read Iris dataset from iris.csv
X = iris_df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values

# Encode the target variable 'species' into numerical labels
class_mapping = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
y = iris_df['species'].map(class_mapping).values

splitter=TrainTestSplit(test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = splitter.split(X, y)
knn_args = {'k': 3}
bagging_clf = Bagging(base_classifier=KNN, n_estimators=10, max_samples=1, base_classifier_args=knn_args)
bagging_clf.fit(X_train, y_train)
y_pred = bagging_clf.predict(X_test)

accu = Accuracy()
accuracy_custom = accu.accuracy_score(y_test, y_pred)

knn_sklearn = KNeighborsClassifier(n_neighbors=3)
bagging_clf_sklearn = BaggingClassifier(estimator=knn_sklearn, n_estimators=10, max_samples=1.0)
bagging_clf_sklearn.fit(X_train, y_train)
y_pred_sklearn = bagging_clf_sklearn.predict(X_test)

accuracy_sklearn = accuracy_score(y_test, y_pred_sklearn)
print(f"Accuracy of Custom bagging is {accuracy_custom}")
print(f"Accuracy of sklearn bagging is {accuracy_sklearn}")
    
