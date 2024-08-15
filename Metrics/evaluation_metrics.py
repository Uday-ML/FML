class evaluation_metrics:

    @staticmethod
    def accuracy(actual_y, predicted_y):
        correct = sum(1 for a, p in zip(actual_y, predicted_y) if a == p)
        total = len(actual_y)
        return correct / total

    @staticmethod
    def precision(actual_y, predicted_y, positive_label=1):
        true_positive = sum(1 for a, p in zip(actual_y, predicted_y) if a == p == positive_label)
        predicted_positive = sum(1 for p in predicted_y if p == positive_label)
        return true_positive / predicted_positive if predicted_positive != 0 else 0

    @staticmethod
    def recall(actual_y, predicted_y, positive_label=1):
        true_positive = sum(1 for a, p in zip(actual_y, predicted_y) if a == p == positive_label)
        actual_positive = sum(1 for a in actual_y if a == positive_label)
        return true_positive / actual_positive if actual_positive != 0 else 0

    @staticmethod
    def f1_score(actual_y, predicted_y, positive_label=1):
        precision_val = evaluation_metrics.precision(actual_y, predicted_y, positive_label)
        recall_val = evaluation_metrics.recall(actual_y, predicted_y, positive_label)
        return 2 * (precision_val * recall_val) / (precision_val + recall_val) if (precision_val + recall_val) != 0 else 0

    @staticmethod
    def confusion_matrix(actual_y, predicted_y):
        true_positive = sum(1 for a, p in zip(actual_y, predicted_y) if a == p == 1)
        true_negative = sum(1 for a, p in zip(actual_y, predicted_y) if a == p == 0)
        false_positive = sum(1 for a, p in zip(actual_y, predicted_y) if a == 0 and p == 1)
        false_negative = sum(1 for a, p in zip(actual_y, predicted_y) if a == 1 and p == 0)
        return true_positive, false_positive, false_negative, true_negative

