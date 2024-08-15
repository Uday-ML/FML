import numpy as np
from activations import SoftmaxActivation
from losses import CategoricalCrossentropyLoss
from activations import EnhancedProbabilisticClassifier

class Dense_Layer:
    def __init__(self, size_input, num_neurons, reg_l1=0, reg_l2=0,
                 bias_l1=0, bias_l2=0):
        self.neuron_weights = self.initialize_parameters(
            size_input, num_neurons)
        self.neuron_biases = np.zeros((1, num_neurons))
        self.reg_l1 = reg_l1
        self.reg_l2 = reg_l2
        self.bias_l1 = bias_l1
        self.bias_l2 = bias_l2

    def initialize_parameters(self, size_input, num_neurons):
        return 0.01 * np.random.randn(size_input, num_neurons)

    def perform_forward_pass(self, input_data, is_training):
        self.cache_input_data = input_data
        self.compute_output(input_data)

    def compute_output(self, input_data):
        self.output_data = np.dot(
            input_data, self.neuron_weights) + self.neuron_biases

    def perform_backward_pass(self, gradients_loss):
        self.compute_gradients(gradients_loss)
        self.apply_regularization()

    def compute_gradients(self, gradients_loss):
        self.weight_gradients = np.dot(self.cache_input_data.T, gradients_loss)
        self.bias_gradients = np.sum(gradients_loss, axis=0, keepdims=True)
        self.input_gradients = np.dot(gradients_loss, self.neuron_weights.T)

    def apply_regularization(self):
        if self.reg_l1 > 0:
            self.weight_gradients += self.reg_l1 * self.l1_regularization_gradient()
        if self.reg_l2 > 0:
            self.weight_gradients += 2 * self.reg_l2 * self.neuron_weights
        if self.bias_l1 > 0:
            self.bias_gradients += self.bias_l1 * \
                self.l1_regularization_gradient(self.neuron_biases)
        if self.bias_l2 > 0:
            self.bias_gradients += 2 * self.bias_l2 * self.neuron_biases

    def l1_regularization_gradient(self, array=None):
        if array is None:
            array = self.neuron_weights
        gradient = np.ones_like(array)
        gradient[array < 0] = -1
        return gradient

#############################

# Revised Input layer class


class Input_Layer:
    # Execute forward pass
    def perform_forward_pass(self, input_data, is_training):
        self.output_data = input_data


#############################


class Neural_Networks:
    """
    A simple neural network class.

    Attributes:
        layers (list): List to store layers in the neural network.
        softmax_output (object): Object for softmax activation layer.

    Methods:
        add_layer: Add a layer to the neural network.
        set_config: Set configuration parameters for the neural network.
        configure: Configure the neural network with input layer, trainable layers, and loss.
        train_model: Train the neural network with specified data and parameters.
        forward_pass: Perform a forward pass through the neural network.
        backward_pass: Perform a backward pass through the neural network.
        evaluate_model: Evaluate the neural network on validation data.
        predict_samples: Make predictions on input samples.
    """

    def __init__(self):
        """Initialize the NeuralNetwork class."""
        self.layers = []
        self.softmax_output = None

    def add_layer(self, layer):
        """Add a layer to the neural network."""
        self.layers.append(layer)

    def set_config(self, *, loss, optimizer, accuracy):
        """Set configuration parameters for the neural network."""
        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = accuracy

    def configure(self):
        """Configure the neural network with input layer, trainable layers, and loss."""
        self.input_layer = Input_Layer()
        layer_count = len(self.layers)
        self.trainable_layers = []

        for i in range(layer_count):
            if i == 0:
                self.layers[i].previous_layer = self.input_layer
                self.layers[i].next_layer = self.layers[i + 1]
            elif i < layer_count - 1:
                self.layers[i].previous_layer = self.layers[i - 1]
                self.layers[i].next_layer = self.layers[i + 1]
            else:
                self.layers[i].previous_layer = self.layers[i - 1]
                self.layers[i].next_layer = self.loss
                self.output_activation = self.layers[i]

            if hasattr(self.layers[i], 'neuron_weights'):
                self.trainable_layers.append(self.layers[i])

        self.loss.remember_trainable_layers(
            trainable_layers=self.trainable_layers)

        if isinstance(self.layers[-1], SoftmaxActivation) and isinstance(self.loss, CategoricalCrossentropyLoss):
            self.softmax_output = EnhancedProbabilisticClassifier()

    def fit(self, X, y, *, epochs=1, batch_size=None, print_freq=0, validation_data=None):
        """Train the neural network with specified data and parameters."""
        self.accuracy.initialize(y)
        train_steps = 1

        if validation_data is not None:
            validation_steps = 1
            X_val, y_val = validation_data

        if batch_size is not None:
            train_steps = len(X) // batch_size
            if train_steps * batch_size < len(X):
                train_steps += 1

            if validation_data is not None:
                validation_steps = len(X_val) // batch_size
                if validation_steps * batch_size < len(X_val):
                    validation_steps += 1

        for epoch in range(1, epochs + 1):
            if print_freq:
                print(f'epoch: {epoch}')
            self.loss.start_new_pass()
            self.accuracy.reset_accumulated_data()

            for step in range(train_steps):
                if batch_size is None:
                    batch_X = X
                    batch_y = y
                else:
                    batch_X = X[step * batch_size:(step + 1) * batch_size]
                    batch_y = y[step * batch_size:(step + 1) * batch_size]

                output = self.forward_pass(batch_X, training=True)
                data_loss, reg_loss = self.loss.calculate_total_loss(
                    output, batch_y, include_regularization=True)
                loss = data_loss + reg_loss
                predictions = self.output_activation.generate_predictions(
                    output)
                acc = self.accuracy.compute_accuracy(predictions, batch_y)
                self.backward_pass(output, batch_y)
                self.optimizer.before_update()
                for layer in self.trainable_layers:
                    self.optimizer.update(layer)
                self.optimizer.after_update()

                if print_freq and (not step % print_freq):
                    print(f'step: {step}, ' +
                          f'acc: {acc:.3f}, ' +
                          f'loss: {loss:.3f} (' +
                          f'data_loss: {data_loss:.3f}, ' +
                          f'reg_loss: {reg_loss:.3f}), ' +
                          f'lr: {self.optimizer.current_learning_rate}')

            epoch_data_loss, epoch_reg_loss = self.loss.calculate_accumulated_loss(
                include_regularization=True)
            epoch_loss = epoch_data_loss + epoch_reg_loss
            epoch_acc = self.accuracy.calculate_accumulated_accuracy()

            if print_freq:
                print(f'training, ' +
                      f'acc: {epoch_acc:.3f}, ' +
                      f'loss: {epoch_loss:.3f} (' +
                      f'data_loss: {epoch_data_loss:.3f}, ' +
                      f'reg_loss: {epoch_reg_loss:.3f}), ' +
                      f'lr: {self.optimizer.current_learning_rate}')

            if validation_data is not None:
                self.loss.start_new_pass()
                self.accuracy.reset_accumulated_data()

                for step in range(validation_steps):
                    if batch_size is None:
                        batch_X = X_val
                        batch_y = y_val
                    else:
                        batch_X = X_val[step *
                                        batch_size:(step + 1) * batch_size]
                        batch_y = y_val[step *
                                        batch_size:(step + 1) * batch_size]

                    output = self.forward_pass(batch_X, training=False)
                    self.loss.calculate_total_loss(output, batch_y)
                    predictions = self.output_activation.generate_predictions(
                        output)
                    self.accuracy.compute_accuracy(predictions, batch_y)

                val_loss = self.loss.calculate_accumulated_loss()
                val_acc = self.accuracy.calculate_accumulated_accuracy()
                if print_freq:
                    print(f'validation, ' +
                          f'acc: {val_acc:.3f}, ' +
                          f'loss: {val_loss:.3f}')

    def forward_pass(self, X, training):
        """Perform a forward pass through the neural network."""
        self.input_layer.perform_forward_pass(X, training)
        for layer in self.layers:
            layer.perform_forward_pass(
                layer.previous_layer.output_data, training)
        return layer.output_data

    def backward_pass(self, output, y):
        """Perform a backward pass through the neural network."""
        if self.softmax_output is not None:
            self.softmax_output.perform_backward_pass(output, y)
            self.layers[-1].input_gradients = self.softmax_output.input_gradients
            for layer in reversed(self.layers[:-1]):
                layer.perform_backward_pass(layer.next_layer.input_gradients)
            return
        self.loss.perform_backward(output, y)
        for layer in reversed(self.layers):
            layer.perform_backward_pass(layer.next_layer.input_gradients)

    def evaluate(self, X_val, y_val, *, batch_size=None):
        """Evaluate the neural network on validation data."""
        validation_steps = 1

        if batch_size is not None:
            validation_steps = len(X_val) // batch_size
            if validation_steps * batch_size < len(X_val):
                validation_steps += 1

        self.loss.start_new_pass()
        self.accuracy.reset_accumulated_data()

        for step in range(validation_steps):
            if batch_size is None:
                batch_X = X_val
                batch_y = y_val
            else:
                batch_X = X_val[step * batch_size:(step + 1) * batch_size]
                batch_y = y_val[step * batch_size:(step + 1) * batch_size]

            output = self.forward_pass(batch_X, training=False)
            self.loss.calculate_total_loss(output, batch_y)
            predictions = self.output_activation.generate_predictions(output)
            self.accuracy.compute_accuracy(predictions, batch_y)

        val_loss = self.loss.calculate_accumulated_loss()
        val_acc = self.accuracy.calculate_accumulated_accuracy()
        print(f'validation, ' +
              f'acc: {val_acc:.3f}, ' +
              f'loss: {val_loss:.3f}')

    def predict_samples(self, X, *, batch_size=None):
        """Make predictions on input samples."""
        prediction_steps = 1

        if batch_size is not None:
            prediction_steps = len(X) // batch_size
            if prediction_steps * batch_size < len(X):
                prediction_steps += 1

        outputs = []

        for step in range(prediction_steps):
            if batch_size is None:
                batch_X = X
            else:
                batch_X = X[step * batch_size:(step + 1) * batch_size]

            batch_output = self.forward_pass(batch_X, training=False)
            outputs.append(batch_output)

        return np.vstack(outputs)
