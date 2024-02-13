import datasets
import nltk
nltk.download('stopwords')

def load_and_preprocess_data():
    dataset = datasets.load_dataset('trec')
    train = dataset['train'][:1000]
    val = dataset['test']

    X_train = preprocess([e.strip() for e in train['text']])
    y_train = train['coarse_label']
    X_val = preprocess([e.strip() for e in val['text']])
    y_val = val['coarse_label']
    
    return X_train, y_train, X_val, y_val

def preprocess(X):
    return [x.lower() for x in X]

