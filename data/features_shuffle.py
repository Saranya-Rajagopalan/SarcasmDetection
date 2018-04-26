import pandas as pd
from sklearn import linear_model
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import random

df = pd.read_csv('features.csv')
data = df


data = data.drop(['hashtag_polarity'], axis=1)

def kfolding(data):
    k = KFold(n_splits = 10, shuffle = True)
    cut = next(k.split(data), None)
    train = data.iloc[cut[0]]
    test = data.iloc[cut[1]]
    return test, train

def LR(data):
    acc = []
    train, test = kfolding(data)
    logreg = linear_model.LogisticRegression(C=1e5)
    logreg.fit(train.drop(['label'], axis=1), train['label'])
    predict = logreg.predict(test.drop(['label'], axis=1))
    acc.append(accuracy_score(predict, test['label']))
    return(float(sum(acc)/len(acc)))

def SVM(data):
    acc = []
    train, test = kfolding(data)
    classifier = SVC(C=0.1, kernel='linear')
    classifier.fit(train.drop(['label'], axis=1), train['label'])
    predict = classifier.predict(test.drop(['label'], axis=1))
    acc.append(accuracy_score(predict, test['label']))
    return (float(sum(acc) / len(acc)))

def NB(data):
    acc = []
    train, test = kfolding(data)
    model = GaussianNB()
    model.fit(train.drop(['label'], axis=1), train['label'])
    predict = model.predict(test.drop(['label'], axis=1))
    acc.append(accuracy_score(predict, test['label']))
    return (float(sum(acc) / len(acc)))

def DT(data):
    acc = []
    train, test = kfolding(data)
    classifier = DecisionTreeClassifier()
    classifier.fit(train.drop(['label'], axis=1), train['label'])
    predict = classifier.predict(test.drop(['label'], axis=1))
    acc.append(accuracy_score(predict, test['label']))
    return (float(sum(acc) / len(acc)))

def NN(data):
    acc = []
    train, test = kfolding(data)
    classifier = MLPClassifier(hidden_layer_sizes=(100, 100, 100), max_iter=1000)
    classifier.fit(train.drop(['label'], axis=1), train['label'])
    predict = classifier.predict(test.drop(['label'], axis=1))
    acc.append(accuracy_score(predict, test['label']))
    return (float(sum(acc) / len(acc)))


accuracies = []
list = ['User mention', 'Exclamation', 'Question mark',
       'Ellipsis', 'Interjection', 'UpperCase', 'RepeatLetters',
       'SentimentScore', 'positive word count', 'negative word count',
       'polarity flip', 'Nouns', 'Verbs', 'PositiveIntensifier',
       'NegativeIntensifier', 'Bigrams', 'Trigram', 'Skipgrams',
       'Emoji Sentiment', 'Passive aggressive count',
       'Emoji_tweet_polarity flip']

guesses = []
for i in range(0,100):
    acc = []
    guess = []
    guess = random.sample(list, 5)
    guesses.append(guess)
    guess.append('label')
    data_guess = data[guess]

    for i in range(1,11):
        acc.append(SVM(data_guess))
    print(acc)
    print(np.mean(acc))
    accuracies.append(np.mean(acc))
Features = pd.DataFrame(guesses)
Accu = pd.DataFrame(accuracies)
combo = pd.concat([Features,Accu], ignore_index = True, axis=1)
combo.to_csv("SVM.csv", header=False, index=True)


