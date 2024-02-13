# Naive-Bayes-Classifier for Question Type Classification

In this small project, I  implement a Naive Bayes classifier for classifying question types, and evaluate the performance of the classifier.

### Setup

I used the TREC question classficiation dataset by [Xin and Roth, 2001](https://www.aclweb.org/anthology/C02-1150)

### Dataset Description

The Text REtrieval Conference (TREC) Question Classification dataset contains 5500 labeled questions in training set and another 500 for test set.

The dataset has 6 coarse class labels and 50 fine class labels. I only used the course class labels.

I used a small sample from the dataset.

### Results
**Model performance when removing stop words:**  precision: 0.44771083470098766, recall: 0.42781423510031624, f1: 0.40205032540376506

**Model performance without removing stop words:**  precision: 0.5644659534152084, recall: 0.5030523705543285, f1: 0.5011024157070146
