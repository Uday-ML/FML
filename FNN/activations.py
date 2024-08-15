import numpy as np

# Revised Softmax activation class
class SoftmaxActivation:

    # Execute forward pass
    def perform_forward_pass(self, input_data, is_training):
        # Remember input values
        self.input_data = input_data

        # Obtain unnormalized probabilities
        exp_values = np.exp(input_data - np.max(input_data, axis=1,
                                                 keepdims=True))

        # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1,
                                            keepdims=True)

        self.output_data = probabilities

    # Execute backward pass
    def perform_backward_pass(self, gradient_values):

        # Create uninitialized array
        self.input_gradients = np.empty_like(gradient_values)

        # Enumerate outputs and gradients
        for index, (single_output, single_gradient) in \
                enumerate(zip(self.output_data, gradient_values)):
            # Flatten output array
            single_output = single_output.reshape(-1, 1)
            # Calculate Jacobian matrix of the output
            jacobian_matrix = np.diagflat(single_output) - \
                              np.dot(single_output, single_output.T)
            # Calculate sample-wise gradient
            # and add it to the array of sample gradients
            self.input_gradients[index] = np.dot(jacobian_matrix,
                                                 single_gradient)

    # Calculate predictions for outputs
    def generate_predictions(self, output_values):
        return np.argmax(output_values, axis=1)

##############################################################################

# Rectified Linear Unit (ReLU) activation
class Activation_RectifiedLinearUnit:

    # Forward propagation
    def perform_forward_pass(self, data_input, is_training):
        # Remembering input values
        self.data_input = data_input
        # Calculating output values from inputs
        self.output_data = np.maximum(0, data_input)

    # Backward propagation
    def perform_backward_pass(self, gradient_values):
        # Creating a copy of values to modify the original variable
        self.input_gradients = gradient_values.copy()

        # Zeroing gradients where input values were negative
        self.input_gradients[self.data_input <= 0] = 0

    # Generate predictions for the activation outputs
    def generate_predictions(self, activation_outputs):
        return activation_outputs

##############################################################################

# Modified Sigmoid activation class
class SigmoidActivation:

    # Forward pass
    def perform_forward_pass(self, input_data, is_training):
        # Save input and calculate/save output
        # of the sigmoid function
        self.input_data = input_data
        self.output_data = 1 / (1 + np.exp(-input_data))

    # Backward pass
    def perform_backward_pass(self, gradient_values):
        # Derivative - calculates from output of the sigmoid function
        self.input_gradients = gradient_values * (1 - self.output_data) * self.output_data

    # Calculate predictions for outputs
    def generate_predictions(self, output_values):
        return (output_values > 0.5) * 1

##############################################################################

# Linear transformation activation
class Activation_LinearTransform:

    # Forward propagation
    def perform_forward_pass(self, data_input, is_training):
        # Store input values
        self.data_input = data_input
        self.output_data = data_input

    # Backward propagation
    def perform_backward_pass(self, gradient_values):
        # Derivative is 1, so 1 * gradient_values = gradient_values (applying the chain rule)
        self.backward_gradient = gradient_values.copy()

    # Generate predictions for the transformed outputs
    def generate_predictions(self, transformed_outputs):
        return transformed_outputs

##############################################################################

# Enhanced probabilistic classifier with integrated Softmax activation
# and cross-entropy loss for optimized backward computation
class EnhancedProbabilisticClassifier:

    # Perform backward computation
    def perform_backward_pass(self, gradient_values, true_labels):

        # Determine the number of data samples
        num_samples = len(gradient_values)

        # If labels are one-hot encoded,
        # convert them to discrete values
        if len(true_labels.shape) == 2:
            true_labels = np.argmax(true_labels, axis=1)

        # Create a copy for safe modification
        self.input_gradients = gradient_values.copy()
        # Compute gradient
        self.input_gradients[range(num_samples), true_labels] -= 1
        # Normalize the gradient
        self.input_gradients = self.input_gradients / num_samples
