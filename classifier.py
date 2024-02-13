import math
from collections import Counter, defaultdict
import nltk

class NaiveBayesClassifier:
    """Code for a bag-of-words Naive Bayes classifier.
    """

    def __init__(self, remove_stops: bool = True) -> None:
        self.remove_stops = remove_stops
        if self.remove_stops:
          self.stop_words = set(nltk.corpus.stopwords.words('english'))
        self.classes = {}

        # the following will be populated after the train() function is called
        self.log_prior: Dict[int, float] = None
        self.likelihoods: Dict[int, Dict[str, float]] = None
        self.vocab_all: Union[Set[str], Dict[str, int]] = None
        self.vocab_by_class: Dict[int, Dict[str, int]] = None

    def train(self, X: List[str], y: List[int]) -> None:
        """
        Train the Naive Bayes classification model.

        Args:
          X: training data
          y: labels

        Returns:
          None (updates class attributes self.vocabulary, self.logprior, self.loglikelihood)
        """

        # no of documents
        N_doc = len(X)

        # Initialize classes and vocabulary
        self.classes = set(y)
        self.vocab_all = set()
        self.vocab_by_class = defaultdict(Counter)
        self.log_prior = {}
        self.likelihoods = defaultdict(lambda: defaultdict(float))

        # logprior
        for c in self.classes:
            N_c = sum(1 for label in y if label == c)
            self.log_prior[c] = math.log(N_c / N_doc)

            # Initialize bigdoc for class c
            bigdoc_c = []
            for doc, label in zip(X, y):
                if label == c:
                    if self.remove_stops:
                      tokens = [word for word in doc.strip().split() if word not in self.stop_words]
                    else:
                      tokens = [word for word in doc.strip().split()]
                    bigdoc_c.extend(tokens)
                    self.vocab_all.update(tokens)

            # vocab_by_class update
            self.vocab_by_class[c].update(bigdoc_c)

        # log-likelihood with Laplace smoothing
        for c in self.classes:
            total_words = sum(self.vocab_by_class[c].values())
            for word in self.vocab_all:
                count_w_c = self.vocab_by_class[c][word]
                self.likelihoods[c][word] = math.log((count_w_c + 1) / (total_words + len(self.vocab_all)))

    def predict(self, doc: str) -> int:
        """
        Return the most likely class for a given document.
        Use the likelihood and log_prior values populated during training

        Returns:
            The most likely class as predicted by the model.
        """
        class_scores = {cls: self.log_prior[cls] for cls in self.classes}
        words = doc.strip().split()
        # Calculate score for each class
        for cls in self.classes:
            for word in words:
                if word in self.vocab_all:  # Only consider words that are in the vocabulary, otherwise ignore
                    class_scores[cls] += self.likelihoods[cls].get(word, 0)

        # Return class with the highest score
        return max(class_scores, key=class_scores.get)

    def predict_all(self, test_docs: List[str]) -> List[int]:
        """
        Predict the class of all documents in the test set.
        This is just a loop over all documents in the test set
        """
        y_pred = [self.predict(doc) for doc in test_docs]
        return y_pred

    @staticmethod
    def evaluate(
        y_pred: List[int], y_true: List[int],
    ) -> Tuple[float, float, float]:
        """
        Calculate a precision, recall, and F1 score for the model
        on a given test set. Use macro averaging for these metrics.

        Args:
            y_pred: Predicted labels
            y_true: Ground truth labels

        Returns:
            (float, float, float)
            The model's precision, recall, and F1 score relative to the
            target class.
        """
        precision_score = 0.0
        recall_score = 0.0
        f1_score = 0.0

        classes = set(y_true)
        # Calculate metrics for each class
        for cls in classes:
            TP = sum((y_pred[i] == cls and y_true[i] == cls) for i in range(len(y_pred)))
            FP = sum((y_pred[i] == cls and y_true[i] != cls) for i in range(len(y_pred)))
            FN = sum((y_pred[i] != cls and y_true[i] == cls) for i in range(len(y_pred)))

            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0

            precision_score+= precision
            recall_score += recall
            f1 = (2 * (precision * recall)) / (precision + recall) if (precision + recall) > 0 else 0
            f1_score += f1

        # Macro average the sums by the number of classes
        num_classes = len(classes)
        precision_score = precision_score / num_classes
        recall_score = recall_score / num_classes
        f1_score = f1_score / num_classes

        return precision_score, recall_score, f1_score