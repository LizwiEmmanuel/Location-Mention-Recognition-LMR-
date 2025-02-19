# crf_training.py

import pandas as pd
import nltk
from nltk.tokenize import TweetTokenizer
from sklearn.model_selection import train_test_split
import sklearn_crfsuite
from sklearn_crfsuite import metrics

# Load dataset
train_file_path = 'Train_1.csv'  # Ensure this file is in the same directory
test_file_path = 'Test.csv'  # Test dataset
output_file_path = 'Predictions.csv'

data = pd.read_csv(train_file_path)
test_data = pd.read_csv(test_file_path)

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
def prepare_data(data, is_test=False):
    sentences = []
    for _, row in data.iterrows():
        tokens = tokenizer.tokenize(row['text'])
        if is_test:
            sentences.append(list(zip(tokens, ['O'] * len(tokens))))
        else:
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

train_sentences = prepare_data(data)
test_sentences = prepare_data(test_data, is_test=True)

# Extract features and labels
X_train = [[word2features(s, i) for i in range(len(s))] for s in train_sentences]
y_train = [[label for _, label in s] for s in train_sentences]
X_test = [[word2features(s, i) for i in range(len(s))] for s in test_sentences]

def extract_locations(tokens, labels):
    locations = []
    current_loc = []
    for token, label in zip(tokens, labels):
        if label == 'B-LOC':
            if current_loc:
                locations.append("".join(current_loc))
                current_loc = []
            current_loc.append(token)
        elif label == 'I-LOC':
            current_loc.append(token)
        else:
            if current_loc:
                locations.append("".join(current_loc))
                current_loc = []
    if current_loc:
        locations.append("".join(current_loc))
    return " ".join(locations)

# Train CRF model
crf = sklearn_crfsuite.CRF(algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True)
crf.fit(X_train, y_train)

# Predictions on test set
y_test_pred = crf.predict(X_test)

# Save predictions to CSV
predictions = []
for i, row in test_data.iterrows():
    tokens = tokenizer.tokenize(row['text'])
    labels = y_test_pred[i] if i < len(y_test_pred) else ['O'] * len(tokens)
    locations = extract_locations(tokens, labels)
    predictions.append([row['ID'], locations])

output_df = pd.DataFrame(predictions, columns=['ID', 'Locations'])
output_df.to_csv(output_file_path, index=False)

print("Predictions saved to Predictions.csv")
