import numpy as np

# AdvancedPerformance class for accuracy computation
class AdvancedPerformance:

    # Determine accuracy based on predictions and actual values
    def compute_accuracy(self, predicted_values, actual_values):

        # Obtain comparison results
        comparisons = self.compare_predictions(predicted_values, actual_values)

        # Calculate accuracy
        accuracy = np.mean(comparisons)

        # Add accumulated sum of matching values and sample count
        self.total_sum += np.sum(comparisons)
        self.total_count += len(comparisons)

        # Return computed accuracy
        return accuracy

    # Calculate accumulated accuracy
    def calculate_accumulated_accuracy(self):

        # Compute accumulated accuracy
        accumulated_accuracy = self.total_sum / self.total_count

        # Return the accumulated accuracy
        return accumulated_accuracy

    # Reset variables for accumulated accuracy
    def reset_accumulated_data(self):
        self.total_sum = 0
        self.total_count = 0






class CustomAccuracyClassification(AdvancedPerformance):

    def __init__(self, *, is_binary=False):
        # Check if the model is designed for binary classification
        self.is_binary = is_binary

    # No initialization is needed for this accuracy calculation
    def initialize(self, ground_truth):
        pass

    # Compares model predictions to the ground truth values
    def compare_predictions(self, predictions, ground_truth):
        # Adjust ground truth for multi-class classification
        if not self.is_binary and len(ground_truth.shape) == 2:
            ground_truth = np.argmax(ground_truth, axis=1)

        # Check if predictions match the ground truth
        return predictions == ground_truth





# Custom Regression Accuracy Calculator
class CustomRegressionAccuracyCalculator(AdvancedPerformance):

    def __init__(self):
        # Initialize precision factor
        self.precision_factor = None

    # Calculate precision factor based on ground truth values
    def initialize_precision_factor(self, ground_truth, reinitialize=False):
        if self.precision_factor is None or reinitialize:
            self.precision_factor = np.std(ground_truth) / 250

    # Compare predictions to the ground truth values
    def compare_predictions(self, predicted_values, ground_truth):
        return np.absolute(predicted_values - ground_truth) < self.precision_factor
