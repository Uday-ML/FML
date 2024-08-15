import numpy as np


class Regression:
    def __init__(self, num_variables, learning_rate=0.01, num_epochs=1000):
        if not isinstance(num_variables, int):
            raise ValueError("Number of variables must be an integer.")
        
        if num_variables < 2:
            raise ValueError("Number of variables must be greater than or equal to 2.")
            
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.weights = np.random.rand(num_variables) 
        self.bias = 0
    
    def scale_features(self, X):
        min_vals = np.min(X, axis=0)
        max_vals = np.max(X, axis=0)
        scaled_X = (X - min_vals) / (max_vals - min_vals)
        return scaled_X
    
    def normalize(self, X):
        means = np.mean(X, axis=0)
        stds = np.std(X, axis=0)
        normalized_X = (X - means) / stds
        return normalized_X
    
    def impute_nan(self, X, impute_value=0):
        nan_columns = np.isnan(X).any(axis=0)
        
        if np.any(nan_columns):
            X[:, nan_columns] = np.where(np.isnan(X[:, nan_columns]), impute_value, X[:, nan_columns])
            
        return X

    def fit(self, X, y):
        num_samples, num_features = X.shape

        if num_features != len(self.weights):
            raise ValueError("Number of features in the input data must match the number of variables.")
        
        X = self.impute_nan(X)
        X=self.normalize(X)
        #X=self.scale_features(X)

        for epoch in range(self.num_epochs):
            predictions = np.dot(X, self.weights) + self.bias

            dw = (1/num_samples) * np.dot(X.T, (predictions - y))
            db = (1/num_samples) * np.sum(predictions - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # if epoch % 100 == 0:
            #     loss = (1/(2*num_samples)) * np.sum((predictions - y)**2)
                #print(f'Epoch {epoch}, Loss: {loss}')

    def predict(self, X, y_mean=None):
        X = self.impute_nan(X)
        X = self.normalize(X)
        
        predictions = np.dot(X, self.weights) + self.bias
        
        if y_mean is not None:
            predictions[np.isnan(predictions)] = y_mean
        
        return predictions

