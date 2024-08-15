from evaluation_metrics import evaluation_metrics

actual_y = [1, 0, 1, 1, 0, 1, 0, 1]
predicted_y = [1, 0, 1, 0, 0, 1, 1, 1]

accuracy = evaluation_metrics.accuracy(actual_y, predicted_y)
precision = evaluation_metrics.precision(actual_y, predicted_y)
recall = evaluation_metrics.recall(actual_y, predicted_y)
f1_score = evaluation_metrics.f1_score(actual_y, predicted_y)
confusion_matrix = evaluation_metrics.confusion_matrix(actual_y, predicted_y)

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1_score:.4f}')
print(f'Confusion Matrix: {confusion_matrix}')
