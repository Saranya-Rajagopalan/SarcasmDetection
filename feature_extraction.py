# Code to extract features from a given dataset(.csv file) and generate a feature list(.csv file)

# Import required libraries

import csv
from emoji import UNICODE_EMOJI
import numpy as np
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Variable Initiailization

non_sarcastic_list = []
sarcastic_list = []
sentiment = []
positive = False
negative = False
inversions = []
punctuation_count = []
interjection_count = []
label = []
negative_count = []
positive_count = []
features = []
upperCase = []
emoji = []
accuracy = []

# Reference Lists

interjections = ['wow', 'haha', 'lol', 'sarcasm', 'rofl', 'lmao', 'sarcastic', 'kidding', 'wtf']
exclude = ['I', 'U.S']
emojis = [':)', ';)', '🤔', '🙈', 'así','bla', 'es','se', '😌', 'ds', '💕','👭', ':-)',':p', '(y)']

j = -1

sid = SentimentIntensityAnalyzer()
ps = PorterStemmer()
lemm = WordNetLemmatizer()

# Reads every tweet in the dataset.csv word by word and extracts features

with open('normal.csv', 'rU', encoding='utf8') as fp:
    nsreader = csv.reader(fp, delimiter = ',')
    for i, line in enumerate(nsreader):
        j += 1
        emoji.append(0)
        upperCase.append(0)
        inversions.append(0)
        interjection_count.append(0)
        punctuation_count.append(0)
        non_sarcastic_list.append(line)
        label.append(0)
        negative_count.append(0)
        positive_count.append(0)
        
        # Generate a separate list of labels
        
        label[j] = int(line[0])
        tweet = line[1]
              
        for words in tweet.split(' '):
            
            # Feature - UpperCase word [which is not an interjection]
            if words.isupper() and words not in exclude and words not in interjections:
                upperCase[j] += 1

            # Feature - Emoji [Compared with a list of Unicodes and common emoticons]
            for e in list(UNICODE_EMOJI.keys()) + emojis:
                emoji[j] += words.count(e)

            # Feature - Interjection ['Word' converted to lower case and compared with the list of common interjections]
            for interj in interjections:
                if words.lower().count(interj):
                    interjection_count[j] += 1

            # Feature - Punctuation [Includes punctuation which influence most sarcastic comments ('!' and '?')]
            punctuation_count[j] = punctuation_count[j] + words.count('!') + words.count('?')
            
            # Feature - Number of Positive / Negative words and change in Polarity 
            sentiment.append((words, sid.polarity_scores(words)))
            ss = sid.polarity_scores(words)
            if ss["neg"] == 1.0:
                negative = True
                negative_count[j] += 1
                if positive:
                    inversions[j] += 1 
                    positive = False
            elif ss["pos"] == 1.0:
                positive = True
                positive_count[j] += 1
                if negative:
                    inversions[j] += 1
                    negative = False

# Create a single list of lists with label and all the extracted features
                    
feature_label = list(zip(label, positive_count, negative_count, inversions, punctuation_count, upperCase,
                          interjection_count, emoji))

# Headers for the new feature list

headers = ["label", "positive_count", "negative_count", "inversions", "punctuations", "upperCase", "interjections", "emoji"]

# Writing headers to the new .csv file

with open("feature_list.csv", "w") as header:
    header = csv.writer(header)
    header.writerow(headers)
    
# Append the feature list to the file
    
with open("feature_list.csv", "a") as feature_csv:
    writer = csv.writer(feature_csv)
    for line in feature_label:
        writer.writerow(line)
