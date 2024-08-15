import numpy as np

class Bagging:
    def __init__(self, base_classifier, n_estimators=10, max_samples=1.0, base_classifier_args=None):
        
        self.base_classifier = base_classifier
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.base_classifier_args = base_classifier_args or {}
        self.models = []

    def fit(self, X, y):
        
        self.models = []
        n_samples, _ = X.shape
        n_subsamples = int(n_samples * self.max_samples)

        for _ in range(self.n_estimators):
            indices = np.random.choice(n_samples, n_subsamples, replace=True)
            X_subset, y_subset = X[indices], y[indices]

            model = self.base_classifier(**self.base_classifier_args)
            model.fit(X_subset, y_subset)
            self.models.append(model)

    def predict(self, X):
        
        predictions = np.array([model.predict(X) for model in self.models])
        y_pred = np.mean(predictions, axis=0) >= 0.5
        return y_pred.astype(int)
