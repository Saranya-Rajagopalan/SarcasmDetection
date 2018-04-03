# -*- coding: utf-8 -*-


import csv
import numpy as np
from sklearn import linear_model, datasets
from sklearn.model_selection import KFold
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.metrics import accuracy_score
from nltk.sentiment.vader import SentimentIntensityAnalyzer

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


interjections = ['wow', 'haha', 'lol', 'sarcasm', 'rofl', 'lmao', 'sarcastic', 'kidding', 'wtf']
j = -1

sid = SentimentIntensityAnalyzer()
ps = PorterStemmer()
lemm = WordNetLemmatizer()
with open('sarcasm_0_serious_1.csv', 'rU') as non_sarcasm:
    nsreader = csv.reader(non_sarcasm, delimiter = ' ')
    for i,line in enumerate(nsreader):
        j += 1
        inversions.append(0)
        interjection_count.append(0)
        punctuation_count.append(0)
        non_sarcastic_list.append(line)
        label.append(0)
        negative_count.append(0)
        positive_count.append(0)
        
        label[j] = line[0][0]
        line[0] = line[0][2:]
        
        for words in line:
            
            #print(words)
            if(words.lower() in interjections):
                interjection_count[j] += 1
            punctuation_count[j] = punctuation_count[j] + words.count('!') + words.count('?')
            sentiment.append((words, sid.polarity_scores(words)))
            ss = sid.polarity_scores(words)
            #print(words, ss)
            if(ss["neg"] == 1.0):
                negative = True
                negative_count[j] += 1
                if(positive):
                    inversions[j] += 1 
                    #print(words)
                    positive = False
            elif(ss["pos"] == 1.0):
                positive = True
                positive_count[j] += 1
                if(negative):
                    inversions[j] += 1
                    #print(words)
                    negative = False
            #if(interjection_count[j] != 0):
            
            #print(line, inversions[j], punctuation_count[j], interjection_count[j])

features = np.asarray(zip(positive_count, negative_count, inversions, punctuation_count, interjection_count))
label = np.asarray(label)

print(features, label)

k = KFold(n_splits = 10, shuffle = True)

for train_idx, test_idx in k.split(features):
    features_train, features_test = features[train_idx], features[test_idx]
    label_train, label_test = label[train_idx], label[test_idx]

logreg = linear_model.LogisticRegression(C=1e5)

logreg.fit(features_train, label_train)


predict = logreg.predict(features_test)

print(accuracy_score(predict, label_test))