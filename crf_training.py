# crf_training.py

import pandas as pd
import nltk
from nltk.tokenize import TweetTokenizer
from sklearn.model_selection import train_test_split
import sklearn_crfsuite
from sklearn_crfsuite import metrics

# Load dataset
file_path = 'Train_1.csv'  # Ensure this file is in the same directory
data = pd.read_csv(file_path)

# Initialize tokenizer
tokenizer = TweetTokenizer(preserve_case=True, strip_handles=True, reduce_len=True)

# Function to extract features
def word2features(sent, i):
    word = sent[i][0]
    features = {
        'word.lower()': word.lower(),
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'prefix-3': word[:3],
        'suffix-3': word[-3:],
    }
    if i > 0:
        word1 = sent[i - 1][0]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.isupper()': word1.isupper(),
        })
    else:
        features['BOS'] = True  # Beginning of Sentence
    if i < len(sent) - 1:
        word1 = sent[i + 1][0]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.isupper()': word1.isupper(),
        })
    else:
        features['EOS'] = True  # End of Sentence
    return features

# Convert text and location to BIO format
def prepare_data(data):
    sentences = []
    for _, row in data.iterrows():
        tokens = tokenizer.tokenize(row['text'])
        locations = row['location'].split() if pd.notnull(row['location']) else []
        bio_tags = ['O'] * len(tokens)
        for loc in locations:
            loc_tokens = tokenizer.tokenize(loc)
            for i in range(len(tokens) - len(loc_tokens) + 1):
                if tokens[i:i+len(loc_tokens)] == loc_tokens:
                    bio_tags[i] = 'B-LOC'
                    for j in range(1, len(loc_tokens)):
                        bio_tags[i + j] = 'I-LOC'
                    break
        sentences.append(list(zip(tokens, bio_tags)))
    return sentences

sentences = prepare_data(data)

# Extract features and labels
X = [[word2features(s, i) for i in range(len(s))] for s in sentences]
y = [[label for _, label in s] for s in sentences]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train CRF model
crf = sklearn_crfsuite.CRF(algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True)
crf.fit(X_train, y_train)

# Predictions
y_pred = crf.predict(X_test)

# Evaluate model
print("F1 Score:", metrics.flat_f1_score(y_test, y_pred, average='weighted'))
print("Classification Report:")
print(metrics.flat_classification_report(y_test, y_pred))
