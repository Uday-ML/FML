import numpy as np


class LR:
    def __init__(self, num_variables, learning_rate=0.01, num_epochs=1000):
        if not isinstance(num_variables, int):
            raise ValueError("Number of variables must be an integer.")

        if num_variables < 2:
            raise ValueError("Number of variables must be greater than or equal to 2.")

        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.weights = np.random.rand(num_variables)
        self.bias = 0

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y):
        num_samples, num_features = X.shape

        if num_features != len(self.weights):
            raise ValueError("Number of features in the input data must match the number of variables.")

        for epoch in range(self.num_epochs):
            linear_output = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(linear_output)

            loss = -(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
            loss = np.mean(loss)

            dw = (1/num_samples) * np.dot(X.T, (predictions - y))
            db = (1/num_samples) * np.sum(predictions - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # if epoch % 100 == 0:
            #     print(f'Epoch {epoch}, Loss: {loss}')

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        predictions = self.sigmoid(linear_output)

        labels = (predictions >= 0.5).astype(int)
        return labels

    def predict_proba(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        predictions = self.sigmoid(linear_output)
        return predictions
