# lmr_bio_annotation.py

import pandas as pd
from nltk.tokenize import TweetTokenizer

# Load the dataset
file_path = 'Train_1.csv'  # Ensure this file is in the same directory as the script
data = pd.read_csv(file_path)

# Initialize TweetTokenizer
tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)

# Function to generate BIO tags
def bio_tagging(text, locations):
    tokens = tokenizer.tokenize(text)
    tags = ['O'] * len(tokens)  # Default to Outside (O) tag

    if pd.notnull(locations):
        loc_list = locations.split()
        for loc in loc_list:
            loc_tokens = tokenizer.tokenize(loc)
            for i in range(len(tokens) - len(loc_tokens) + 1):
                if tokens[i:i+len(loc_tokens)] == loc_tokens:
                    tags[i] = 'B-LOC'  # Begin tag
                    for j in range(1, len(loc_tokens)):
                        tags[i + j] = 'I-LOC'  # Inside tag
                    break
    
    return list(zip(tokens, tags))

# Apply BIO tagging
data['bio_tags'] = data.apply(lambda row: bio_tagging(row['text'], row['location']), axis=1)

# Display sample BIO-tagged data
print("\nSample BIO-tagged Data:")
print(data[['text', 'bio_tags']].head())
