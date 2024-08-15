# FML-Library
FML (Foundation of Machine Learning) Library is a Python library that provides a collection of popular machine learning algorithms along with utility functions for seamless integration into your projects. With easy installation using pip, you can quickly leverage the power of machine learning for your data analysis and predictive modeling tasks.

# Installation
You can install FML Library using pip:

pip install fml-library


# Algorithms Included
Linear Regression

Module: fml-library.linear_regression
Example:
from fml-library.linear_regression import linear_regression

# Create linear regression model
model = linear_regression()

# Fit the model with data
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
K-Nearest Neighbors (KNN)

Module: fml-library.knn
Example:
from fml-library.knn import knn

# Create KNN model
model = knn()

# Fit the model with data
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
Naive Bayes

Module: fml-library.naive_bayes
Example:
from fml-library.naive_bayes import naive_bayes

# Create Naive Bayes model
model = naive_bayes()

# Fit the model with data
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
Support Vector Machine (SVM)

Module: fml-library.svm
Example:
from fml-library.svm import svm

# Create SVM model
model = svm()

# Fit the model with data
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
Decision Tree

Module: fml-library.decision_tree
Example:
from fml-library.decision_tree import decision_tree

# Create Decision Tree model
model = decision_tree()

# Fit the model with data
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
Logistic Regression

Module: fml-library.logistic_regression_classifier
Example:
from fml-library.logistic_regression_classifier import logistic_regression

# Create Logistic Regression model
model = logistic_regression()

# Fit the model with data
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
Neural Network

Module: fml-library.neural_network
Example:
from fml-library.neural_network import neural_network

# Create Neural Network model
model = neural_network()

# Fit the model with data
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
Utility Functions

Module: fml-library.utility_functions
Includes various utility functions for data preprocessing, feature engineering, and evaluation metrics.
Example Usage
# Import the desired algorithm
from fml-library.linear_regression import linear_regression

# Create the model
model = linear_regression()

# Fit the model with training data
model.fit(X_train, y_train)

# Make predictions on test data
predictions = model.predict(X_test)

# Evaluate the model
accuracy = model.evaluate(X_test, y_test)
print(f"Model Accuracy: {accuracy}")
Contributing
If you find any issues or have suggestions for improvements, please feel free to open an issue or submit a pull request. We welcome contributions from the community!

