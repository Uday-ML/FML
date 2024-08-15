import numpy as np

# Revised Common loss class
class CustomLoss:

    def __init__(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0
        self.trainable_layers = []

    # Regularization loss calculation
    def calculate_regularization_loss(self):

        # 0 by default
        regularization_loss = 0

        # Calculate regularization loss
        # iterate all trainable layers
        for layer in self.trainable_layers:

            # L1 regularization - weights
            # calculate only when factor greater than 0
            if layer.reg_l1 > 0:
                regularization_loss += layer.reg_l1 * \
                                       np.sum(np.abs(layer.neuron_weights))

            # L2 regularization - weights
            if layer.reg_l2 > 0:
                regularization_loss += layer.reg_l2 * \
                                       np.sum(layer.neuron_weights * layer.neuron_weights)

            # L1 regularization - biases
            # calculate only when factor greater than 0
            if layer.bias_l1 > 0:
                regularization_loss += layer.bias_l1 * \
                                       np.sum(np.abs(layer.neuron_biases))

            # L2 regularization - biases
            if layer.bias_l2 > 0:
                regularization_loss += layer.bias_l2 * \
                                       np.sum(layer.neuron_biases * layer.neuron_biases)

        return regularization_loss

    # Set/remember trainable layers
    def remember_trainable_layers(self, trainable_layers):
        self.trainable_layers = trainable_layers

    # Calculates the data and regularization losses
    # given model output and ground truth values
    def calculate_total_loss(self, output, y, include_regularization=False):

        # Calculate sample losses
        sample_losses = self.calculate_forward(output, y)

        # Calculate mean loss
        data_loss = np.mean(sample_losses)

        # Add accumulated sum of losses and sample count
        self.accumulated_sum += np.sum(sample_losses)
        self.accumulated_count += len(sample_losses)

        # If just data loss - return it
        if not include_regularization:
            return data_loss

        # Return the data and regularization losses
        return data_loss, self.calculate_regularization_loss()

    # Calculates accumulated loss
    def calculate_accumulated_loss(self, include_regularization=False):

        # Calculate mean loss
        data_loss = self.accumulated_sum / self.accumulated_count

        # If just data loss - return it
        if not include_regularization:
            return data_loss

        # Return the data and regularization losses
        return data_loss, self.calculate_regularization_loss()

    # Reset variables for accumulated loss
    def start_new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0

    # Forward pass method (needs to be implemented in child classes)
    def calculate_forward(self, output, y):
        raise NotImplementedError("Forward method must be implemented in the child class.")

######################################

# Revised Binary Cross-entropy loss class
class BinaryCrossentropyLoss(CustomLoss):

    # Forward pass
    def calculate_forward(self, predicted_values, true_values):

        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        predicted_values_clipped = np.clip(predicted_values, 1e-7, 1 - 1e-7)
        # Calculate sample-wise loss
        sample_losses = -(true_values * np.log(predicted_values_clipped) +
                          (1 - true_values) * np.log(1 - predicted_values_clipped))
        sample_losses = np.mean(sample_losses, axis=-1)

        # Return losses
        return sample_losses

    # Backward pass
    def perform_backward(self, gradient_values, true_values):

        # Number of samples
        num_samples = len(gradient_values)
        # Number of outputs in every sample
        # We'll use the first sample to count them
        num_outputs = len(gradient_values[0])

        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        clipped_gradient_values = np.clip(gradient_values, 1e-7, 1 - 1e-7)

        # Calculate gradient
        self.input_gradients = -(true_values / clipped_gradient_values -
                                (1 - true_values) / (1 - clipped_gradient_values)) / num_outputs
        # Normalize gradient
        self.input_gradients = self.input_gradients / num_samples

######################################

# Revised Mean Squared Error loss class
class MeanSquaredErrorLoss(CustomLoss):  # L2 loss

    # Forward pass
    def calculate_forward(self, predicted_values, true_values):

        # Calculate loss
        sample_losses = np.mean((true_values - predicted_values)**2, axis=-1)

        # Return losses
        return sample_losses

    # Backward pass
    def perform_backward(self, gradient_values, true_values):

        # Number of samples
        num_samples = len(gradient_values)
        # Number of outputs in every sample
        # We'll use the first sample to count them
        num_outputs = len(gradient_values[0])

        # Gradient on values
        self.input_gradients = -2 * (true_values - gradient_values) / num_outputs
        # Normalize gradient

######################################

# Revised Mean Absolute Error loss class
class MeanAbsoluteErrorLoss(CustomLoss):  # L1 loss

    # Forward pass
    def calculate_forward(self, predicted_values, true_values):

        # Calculate loss
        sample_losses = np.mean(np.abs(true_values - predicted_values), axis=-1)

        # Return losses
        return sample_losses

    # Backward pass
    def perform_backward(self, gradient_values, true_values):

        # Number of samples
        num_samples = len(gradient_values)
        # Number of outputs in every sample
        # We'll use the first sample to count them
        num_outputs = len(gradient_values[0])

        # Calculate gradient
        self.gradient_inputs = np.sign(true_values - gradient_values) / num_outputs
        # Normalize gradient
        self.gradient_inputs = self.gradient_inputs / num_samples

##############################################################################

# Revised Cross-entropy loss class
class CategoricalCrossentropyLoss(CustomLoss):

    # Forward pass
    def calculate_forward(self, predicted_values, true_values):

        # Number of samples in a batch
        num_samples = len(predicted_values)

        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        clipped_predictions = np.clip(predicted_values, 1e-7, 1 - 1e-7)

        # Probabilities for target values -
        # only if categorical labels
        # print(clipped_predictions)
        if len(true_values.shape) == 1:
            correct_confidences = clipped_predictions[range(num_samples),true_values]

        # Mask values - only for one-hot encoded labels
        elif len(true_values.shape) == 2:
            correct_confidences = np.sum(
                clipped_predictions * true_values,
                axis=1
            )

        # Losses
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    # Backward pass
    def perform_backward(self, gradient_values, true_values):

        # Number of samples
        num_samples = len(gradient_values)
        # Number of labels in every sample
        # We'll use the first sample to count them
        num_labels = len(gradient_values[0])

        # If labels are sparse, turn them into one-hot vector
        if len(true_values.shape) == 1:
            true_values = np.eye(num_labels)[true_values]

        # Calculate gradient
        self.input_gradients = -true_values / gradient_values
        # Normalize gradient
        self.input_gradients = self.input_gradients / samples
