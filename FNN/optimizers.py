import numpy as np


class AdamOptimizer:

    def __init__(self, learning_rate=0.1, decay=0.0, epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        # Initialize the Adam optimizer with specified learning rate, decay, epsilon, and beta values
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def before_update(self):
        # Adjust learning rate with decay if specified
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1.0 + self.decay * self.iterations))

    def update(self, layer):
        # Initialize momentums and cache arrays if not present in the layer
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.neuron_weights)
            layer.weight_cache = np.zeros_like(layer.neuron_weights)
            layer.bias_momentums = np.zeros_like(layer.neuron_biases)
            layer.bias_cache = np.zeros_like(layer.neuron_biases)

        # Update momentums with current gradients
        layer.weight_momentums = self.beta_1 * layer.weight_momentums + \
            (1 - self.beta_1) * layer.weight_gradients
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + \
            (1 - self.beta_1) * layer.bias_gradients

        # Get corrected momentums
        weight_momentums_corrected = layer.weight_momentums / \
            (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / \
            (1 - self.beta_1 ** (self.iterations + 1))

        # Update cache with squared current gradients
        layer.weight_cache = self.beta_2 * layer.weight_cache + \
            (1 - self.beta_2) * layer.weight_gradients**2
        layer.bias_cache = self.beta_2 * layer.bias_cache + \
            (1 - self.beta_2) * layer.bias_gradients**2

        # Get corrected cache
        weight_cache_corrected = layer.weight_cache / \
            (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / \
            (1 - self.beta_2 ** (self.iterations + 1))

        # Update weights using Adam formula
        layer.neuron_weights += -self.current_learning_rate * \
            weight_momentums_corrected / \
            (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.neuron_biases += -self.current_learning_rate * \
            bias_momentums_corrected / \
            (np.sqrt(bias_cache_corrected) + self.epsilon)

    def after_update(self):
        # Increment iteration count after parameter update
        self.iterations += 1


class AdagradOptimizer:

    def __init__(self, learning_rate=1.0, decay=0.0, epsilon=1e-7):
        # Initialize the Adagrad optimizer with specified learning rate, decay, and epsilon
        
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon

    def before_update(self):
        # Adjust learning rate with decay if specified
        if self.decay:
            self.current_learning_rate = self.learning_rate / (1.0 + self.decay * self.iterations)

    def update(self, layer):
        # Initialize cache arrays if not present in the layer
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.neuron_weights)
            layer.bias_cache = np.zeros_like(layer.neuron_biases)

        # Update cache with squared gradients
        layer.weight_cache += layer.weight_gradients**2
        layer.bias_cache += layer.bias_gradients**2

        # Update weights using Adagrad formula
        layer.neuron_weights += -self.current_learning_rate * layer.weight_gradients / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.neuron_biases += -self.current_learning_rate * layer.bias_gradients / (np.sqrt(layer.bias_cache) + self.epsilon)

    def after_update(self):
        # Increment iteration count after parameter update
        self.iterations += 1


class SGDOptimizer:
    def __init__(self, learning_rate=1.0, decay=0.0, momentum=0.0):
        # Initialize the SGD optimizer with specified learning rate, decay, and momentum values
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    def before_update(self):
        # Adjust learning rate with decay if specified
        if self.decay:
            self.current_learning_rate = self.learning_rate / \
                (1.0 + self.decay * self.iterations)

    def update(self, layer):
        # If momentum is used
        if self.momentum:
            # If layer does not contain momentum arrays, create them filled with zeros
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.neuron_weights)
                layer.bias_momentums = np.zeros_like(layer.neuron_biases)

            # Update momentum arrays using current gradients
            layer.weight_momentums = self.momentum * layer.weight_momentums - \
                self.current_learning_rate * layer.weight_gradients
            layer.bias_momentums = self.momentum * layer.bias_momentums - \
                self.current_learning_rate * layer.bias_gradients

            # Update weights and biases with momentum
            layer.neuron_weights += layer.weight_momentums
            layer.neuron_biases += layer.bias_momentums
        else:
            # Update weights and biases without momentum
            layer.neuron_weights += -self.current_learning_rate * layer.weight_gradients
            layer.neuron_biases += -self.current_learning_rate * layer.bias_gradients

    def after_update(self):
        # Increment iteration count after parameter update
        self.iterations += 1






class RMSpropOptimizer:

    def __init__(self, learning_rate=0.001, decay=0.0, epsilon=1e-7, rho=0.9):
        # Initialize the RMSprop optimizer with specified learning rate, decay, epsilon, and rho values
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho

    def before_update(self):
        # Adjust learning rate with decay if specified
        if self.decay:
            self.current_learning_rate = self.learning_rate / (1.0 + self.decay * self.iterations)

    def update(self, layer):
        # Initialize cache arrays if not present in the layer
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.neuron_weights)
            layer.bias_cache = np.zeros_like(layer.neuron_biases)

        # Update cache with squared current gradients using RMSprop formula
        layer.weight_cache = self.rho * layer.weight_cache + (1 - self.rho) * layer.weight_gradients**2
        layer.bias_cache = self.rho * layer.bias_cache + (1 - self.rho) * layer.bias_gradients**2

        # Update weights using RMSprop formula
        layer.neuron_weights += -self.current_learning_rate * layer.weight_gradients / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.neuron_biases += -self.current_learning_rate * layer.bias_gradients / (np.sqrt(layer.bias_cache) + self.epsilon)

    def after_update(self):
        # Increment iteration count after parameter update
        self.iterations += 1




