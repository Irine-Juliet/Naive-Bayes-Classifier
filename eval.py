def evaluate_model(y_pred, y_true):
    """Returns precision, recall, and F1 score for model predictions."""
    precision, recall, f1 = NaiveBayesClassifier.evaluate(y_pred, y_true)
    return precision, recall, f1