import pandas as pd
import csv
from sklearn.decomposition import PCA
import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold


features = []
label = []
headers = []

loop_accuracy = []

FEATURE_LIST_CSV_FILE_PATH = os.curdir + "\\data\\feature_list.csv"

def read_data(filename):
    with open(filename, 'rU') as fp:
        nsreader = csv.reader(fp, delimiter = ',')
        for i, line in enumerate(nsreader):
            if i == 0:
                headers = line
            else:
                features.append(line[1:])
                label.append(line[0])
                
    return features, label

features, label = read_data(FEATURE_LIST_CSV_FILE_PATH)
#print(features)


features = np.asarray(features, dtype='float')
label = np.asarray(label, dtype='float')



pca_list = [0.15, 0.25, 0.35, 0.45, 0.65, 0.75, 0.95, 1]

# Using Sklearn Logistic Regression Model for classification of Sarcastic and Non-Sarcastic Tweets
for n in pca_list:
    
    accuracy = []
    k = KFold(n_splits = 10, shuffle = True)
    print(n)
    for train_idx, test_idx in k.split(features):
    
        predict = []
        
        train_features, test_features = features[train_idx], features[test_idx]
        train_lbl, test_lbl = label[train_idx], label[test_idx]
        
    
        
        train_features = StandardScaler().fit_transform(train_features)
        test_features = StandardScaler().fit_transform(test_features)
        pca = PCA(n)
    
    
        
        pca.fit(train_features)
        
        
        train_features = pca.fit_transform(train_features)
        pca.fit(test_features)
        
        test_features = pca.fit_transform(test_features)
        #
        LR = LogisticRegression(solver = 'lbfgs')
        #
        LR.fit(train_features, train_lbl)
        #
      
        predict = LR.predict(test_features)
        
        accuracy.append(accuracy_score(predict, test_lbl))
        
    
    loop_accuracy.append(float(sum(accuracy)/len(accuracy)))


plt.plot(loop_accuracy)