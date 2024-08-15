import numpy as np

class TrainTestSplit:
    def __init__(self, test_size=0.2, random_state=None):
        self.test_size = test_size
        self.random_state = random_state

    def split(self, X, y):
        if X.shape[0] != len(y):
            raise ValueError("The number of datapoints in X must be equal to the number of labels in y!")

        if self.random_state is not None:
            np.random.seed(self.random_state)

        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)

        test_size = int(self.test_size * X.shape[0])
        test_indices, train_indices = indices[:test_size], indices[test_size:]

        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]

        return X_train, X_test, y_train, y_test

class Accuracy:
    @staticmethod
    def accuracy_score(y_true, y_pred):
        if len(y_true) != len(y_pred):
            raise ValueError("Input arrays must have the same length.")

        correct_predictions = np.sum(y_true == y_pred)
        total_samples = len(y_true)

        accuracy = correct_predictions / total_samples
        return accuracy
    
class MeanSquaredError:
    def __init__(self):
        pass

    @staticmethod
    def calculate(y_true, y_pred):
        if len(y_true) != len(y_pred):
            raise ValueError("Input arrays must have the same length.")
        
        squared_errors = [(true - pred) ** 2 for true, pred in zip(y_true, y_pred)]
        mean_squared_error = sum(squared_errors) / len(y_true)
        
        return mean_squared_error