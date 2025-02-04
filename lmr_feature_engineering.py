# lmr_feature_engineering.py

import pandas as pd
import re
from nltk.tokenize import TweetTokenizer
from nltk import pos_tag
from gensim.models import Word2Vec

# Load the dataset
file_path = 'Train_1.csv'  # Ensure this file is in the same directory as the script
data = pd.read_csv(file_path)

# Initialize Tokenizer
tokenizer = TweetTokenizer(preserve_case=True, strip_handles=True, reduce_len=True)

# Function to extract lexical features
def lexical_features(tokens):
    features = []
    for token in tokens:
        feat = {
            'token': token,
            'is_capitalized': token[0].isupper(),
            'is_all_caps': token.isupper(),
            'is_all_lower': token.islower(),
            'prefix-3': token[:3],
            'suffix-3': token[-3:],
            'word_shape': re.sub('[a-z]', 'x', re.sub('[A-Z]', 'X', re.sub('[0-9]', 'd', token)))
        }
        features.append(feat)
    return features

# Function to extract contextual features
def contextual_features(tokens):
    pos_tags = pos_tag(tokens)
    features = []
    for i, (token, pos) in enumerate(pos_tags):
        context = {
            'token': token,
            'pos': pos,
            'prev_token': tokens[i-1] if i > 0 else '<START>',
            'next_token': tokens[i+1] if i < len(tokens)-1 else '<END>'
        }
        features.append(context)
    return features

# Embedding model (Word2Vec)
tokenized_tweets = data['text'].dropna().apply(tokenizer.tokenize)
w2v_model = Word2Vec(sentences=tokenized_tweets, vector_size=100, window=5, min_count=1, workers=4)

# Function to get embedding features
def embedding_features(tokens):
    features = []
    for token in tokens:
        if token in w2v_model.wv:
            embedding = w2v_model.wv[token]
        else:
            embedding = [0] * 100  # Zero vector for OOV words
        features.append({'token': token, 'embedding': embedding})
    return features

# Combine all features
def extract_features(text):
    tokens = tokenizer.tokenize(text)
    lexical = lexical_features(tokens)
    contextual = contextual_features(tokens)
    embedding = embedding_features(tokens)

    combined_features = []
    for lex, ctx, emb in zip(lexical, contextual, embedding):
        combined = {**lex, **ctx, **emb}
        combined_features.append(combined)
    return combined_features

# Apply feature extraction
data['features'] = data['text'].dropna().apply(extract_features)

# Display sample feature-engineered data
print("\nSample Feature-Engineered Data:")
print(data[['text', 'features']].head())
