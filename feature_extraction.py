# -*- coding: utf-8 -*-


import csv
from emoji import UNICODE_EMOJI
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from nltk.sentiment.vader import SentimentIntensityAnalyzer


negative_count = []
positive_count = []
sentiment = []
inversions = []
punctuation_count = []
interjection_count = []
upperCase = []
emoji = []
features = []

label = []
accuracy = []

interjections = ['wow', 'haha', 'lol', 'sarcasm', 'rofl', 'lmao', 'sarcastic', 'kidding', 'wtf', 'if only',
                 'thanks to']
exclude = ['I', 'U.S']
emojis = [':)', ';)', 'ðŸ¤”', 'ðŸ™ˆ', 'asÃ­', 'bla', 'es', 'se', 'ðŸ˜Œ', 'ds', 'ðŸ’•', 'ðŸ‘­', ':-)', ':p']


sid = SentimentIntensityAnalyzer()

with open('B:\\MyCodebase\\SarcasmDetection\\data\\sarcasm_0_serious_1.csv', 'rU', encoding='utf-8') as non_sarcasm:
    nsreader = csv.reader(non_sarcasm, delimiter=' ')
    positive, negative = False, False
    for j, line in enumerate(nsreader):
        emoji.append(0)
        upperCase.append(0)
        inversions.append(0)
        interjection_count.append(0)
        punctuation_count.append(0)
        label.append(0)
        negative_count.append(0)
        positive_count.append(0)

        label[j] = line[0][0]
        line[0] = line[0][2:]

        for words in line:
            if (words.isupper() and words not in exclude and words not in interjections):
                upperCase[j] += 1
            if (words in UNICODE_EMOJI or words in emojis):
                emoji[j] += 1
            if (words.lower() in interjections):
                interjection_count[j] += 1
            punctuation_count[j] = punctuation_count[j] + words.count('!') + words.count('?')
            sentiment.append((words, sid.polarity_scores(words)))
            ss = sid.polarity_scores(words)
            if (ss["neg"] == 1.0):
                negative = True
                negative_count[j] += 1
                if (positive):
                    inversions[j] += 1
                    positive = False
            elif (ss["pos"] == 1.0):
                positive = True
                positive_count[j] += 1
                if (negative):
                    inversions[j] += 1
                    negative = False


features_list = [x for x in zip(positive_count, negative_count, inversions, punctuation_count,
                          interjection_count, emoji)]

features = np.asarray(features_list)
label = np.asarray(label)

k = KFold(n_splits=10, shuffle=True)

for train_idx, test_idx in k.split(features):
    features_train, features_test = features[train_idx], features[test_idx]
    label_train, label_test = label[train_idx], label[test_idx]

    logreg = linear_model.LogisticRegression(C=1e5)

    logreg.fit(features_train, label_train)

    predict = logreg.predict(features_test)

    print(accuracy_score(predict, label_test))
    accuracy.append(accuracy_score(predict, label_test))

print(float(sum(accuracy) / len(accuracy)))
