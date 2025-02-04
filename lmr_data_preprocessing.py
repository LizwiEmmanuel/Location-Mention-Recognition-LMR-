# lmr_data_preprocessing.py

import pandas as pd
import re
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk import pos_tag
import emoji

# Load the dataset
file_path = 'Train_1.csv'  # Ensure this file is in the same directory as the script
data = pd.read_csv(file_path)

# Filter out rows with non-null text
tweets = data['text'].dropna()

# Initialize TweetTokenizer
tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)

# Function to clean text
def clean_text(text):
    text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
    text = emoji.replace_emoji(text, replace='')  # Remove emojis
    text = re.sub(r'[^a-zA-Z0-9\s#@]', '', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
    return text

# Function to handle common slang and abbreviations
slang_dict = {
    "lol": "laughing out loud",
    "omg": "oh my god",
    "brb": "be right back",
    "idk": "i do not know",
    "smh": "shaking my head",
    "btw": "by the way",
    "fyi": "for your information",
    "lmao": "laughing my ass off"
}

def handle_slang(text):
    words = text.split()
    return ' '.join([slang_dict.get(word, word) for word in words])

# Function for tokenization and POS tagging
def preprocess_tweet(text):
    cleaned = clean_text(text)
    normalized = handle_slang(cleaned)
    tokens = tokenizer.tokenize(normalized)
    pos_tags = pos_tag(tokens)
    return tokens, pos_tags

# Apply preprocessing
data['tokens_pos'] = data['text'].dropna().apply(preprocess_tweet)

# Display sample processed data
print("\nSample Processed Data:")
print(data[['text', 'tokens_pos']].head())
