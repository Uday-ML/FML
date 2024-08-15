from sklearn.model_selection import train_test_split
import numpy as np
import nnfs
from nnfs.datasets import vertical_data, spiral_data

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from nnfs.datasets import vertical_data, spiral_data

from neural_networks import *
from activations import *
from losses import *
from metrics import *
from optimizers import *

# Assume you have your features in X and labels in y
# X and y should be NumPy arrays or pandas DataFrames/Series
n_samples = 10000
n_classes = 3
# set dataset to 'vertical_data' or 'spiral_data'
dataset = 'spiral_data'
if dataset == 'spiral_data':
    X, y = spiral_data(n_samples, n_classes)
else:
    X, y = vertical_data(n_samples, n_classes)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Standardize the features (optional but often recommended for neural networks)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the MLPClassifier
# Adjust hyperparameters as needed

model = Neural_Networks()
model.add_layer(Dense_Layer(X_train_scaled.shape[1], 16))
model.add_layer(Activation_RectifiedLinearUnit())
model.add_layer(Dense_Layer(16, 4))
model.add_layer(Activation_RectifiedLinearUnit())
model.add_layer(Dense_Layer(4, n_classes))
model.add_layer(SoftmaxActivation())

# Set loss, optimizer and accuracy objects
model.set_config(
    loss=CategoricalCrossentropyLoss(),
    optimizer=SGDOptimizer(learning_rate=0.3, decay=1e-7, momentum=1e-1),
    accuracy=CustomAccuracyClassification()
)

# Finalize the model
model.configure()

# Train the model
model.fit(X_train_scaled, y_train,
          epochs=2000, batch_size=1000, print_freq=0)

model.evaluate(X_test_scaled, y_test)

y_pred = np.argmax(model.predict_samples(X_test_scaled), axis=1)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

