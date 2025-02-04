# lmr_data_analysis.py

import pandas as pd
import re
from langdetect import detect
from collections import Counter

# Load the dataset
file_path = 'Train_1.csv'  # Ensure this file is in the same directory as the script
data = pd.read_csv(file_path)

# Display basic information about the dataset
print("Dataset Information:")
print(data.info())
print("\nSample Data:")
print(data.head())

# Filter out rows with non-null text for analysis
tweets = data['text'].dropna()

# 1. Average tweet length
avg_length = tweets.apply(len).mean()

# 2. Presence of hashtags, mentions, and URLs
hashtags_count = tweets.apply(lambda x: len(re.findall(r'#\w+', x))).sum()
mentions_count = tweets.apply(lambda x: len(re.findall(r'@\w+', x))).sum()
urls_count = tweets.apply(lambda x: len(re.findall(r'http\S+|www\S+', x))).sum()

# 3. Detect multilingual content (sample 100 tweets for efficiency)
sample_tweets = tweets.sample(100, random_state=42)
languages = sample_tweets.apply(lambda x: detect(x) if pd.notnull(x) else 'unknown')
language_distribution = Counter(languages)

# 4. Common slang and abbreviations (example set)
slang_words = ['lol', 'omg', 'brb', 'idk', 'smh', 'btw', 'fyi', 'lmao']
slang_count = tweets.apply(lambda x: sum(1 for word in slang_words if re.search(r'\\b' + word + r'\\b', x.lower()))).sum()

# Display the analysis results
print("\nAnalysis Results:")
print(f"Average Tweet Length: {avg_length:.2f} characters")
print(f"Total Hashtags: {hashtags_count}")
print(f"Total Mentions: {mentions_count}")
print(f"Total URLs: {urls_count}")
print(f"Language Distribution (Sample of 100 Tweets): {language_distribution}")
print(f"Total Slang Words Detected: {slang_count}")
