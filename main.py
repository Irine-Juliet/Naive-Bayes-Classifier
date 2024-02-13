from data_loader import load_and_preprocess_data
from naive_bayes_classifier import NaiveBayesClassifier
from evaluation import evaluate_model
from dataset_inspector import print_data_sample, reversed_label_mappings 

def load_data_for_inspection():
    dataset = datasets.load_dataset('trec')
    return dataset

def main():
    # Load data for inspection
    dataset_for_inspection = load_data_for_inspection()
    print("Inspecting raw dataset sample:")
    print_data_sample(dataset_for_inspection['train'], 'text', 'coarse_label', print_count=3, label_mappings=reversed_label_mappings)

    X_train, y_train, X_val, y_val = load_and_preprocess_data()
    
    clf = NaiveBayesClassifier(remove_stops=True)
    clf.train(X_train, y_train)
    
    y_pred = clf.predict_all(X_val)
    precision, recall, f1 = evaluate_model(y_pred, y_val)
    
    print(f'precision: {precision}, recall: {recall}, f1: {f1}')

if __name__ == "__main__":
    main()

