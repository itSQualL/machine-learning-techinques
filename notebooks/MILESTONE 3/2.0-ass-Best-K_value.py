#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 17:46:51 2018
@author: alberto
"""


from sklearn import metrics
from sklearn import neighbors

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("../../data/source/Moriarty2.csv",
                  usecols=["UUID","ActionType"])

df2 = pd.read_csv("/home/alberto/Documentos/MatchingLearning/Practicas/T4.csv",
                  usecols=["UUID", "CPU_0", "CPU_1", "CPU_2", "CPU_3", "Traffic_TotalRxBytes", "Traffic_TotalTxBytes", "MemFree"])

df['UUID'] = pd.to_datetime(df['UUID'], unit="ms")
df['UUID'] = df['UUID'].dt.round('t')


df2['UUID'] = pd.to_datetime(df['UUID'], unit="ms")
df2['UUID'] = df['UUID'].dt.round('t')

data = pd.merge(df,df2, on=['UUID'])

data['ActionType'] = data['ActionType'].replace(['malicious'], 1)
data['ActionType'] = data['ActionType'].replace(['benign'], 0)
data = data.drop('UUID', 1)

y = data['ActionType']
data = data.drop('ActionType', 1)


from sklearn import preprocessing
scaler = preprocessing.StandardScaler()
datanorm = scaler.fit_transform(data)

from sklearn.decomposition import PCA
n_components = 1
estimator = PCA(n_components)
X_pca = estimator.fit_transform(datanorm)

x = X_pca


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)




#from sklearn.model_selection import KFold
#
#kf = KFold(n_splits=2)
#for train, test in kf.split(x):
#    print("%s %s" % (train, test))

for i in range(1,30):
    knn=neighbors.KNeighborsClassifier(n_neighbors=i)
    modelKNN=knn.fit(X_train, y_train)
    predKNN=modelKNN.predict(X_test)
    w= metrics.accuracy_score(predKNN,y_test)
    print(i)
print(w)
