#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 15:28:11 2018

@author: alberto
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   

df = pd.read_csv("/home/alberto/Documentos/MatchingLearning/Practicas/Moriarty2.csv",
                  usecols=["UUID","ActionType"])

df2 = pd.read_csv("/home/alberto/Documentos/MatchingLearning/Practicas/T4.csv", 
                  usecols=["UUID", "Total_CPU", "Traffic_TotalTxBytes", "processes", "MemFree", "SwapFree"])

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
data



from sklearn import preprocessing
scaler = preprocessing.StandardScaler()
datanorm = scaler.fit_transform(data)

from sklearn.decomposition import PCA
n_components = 3
estimator = PCA(n_components)
X_pca = estimator.fit_transform(datanorm)
#print (estimator.explained_variance_ratio_)
#print(estimator.explained_variance_ratio_.cumsum())


#Plotting PCA.


fig = plt.figure()
ax = Axes3D(fig)
x_pca = X_pca[:,0]
y_pca = X_pca[:,1]
z_pca = X_pca[:,2]
ax.scatter(x_pca,y_pca,z_pca, c=y.values)
plt.show()


x = X_pca


from sklearn.metrics import accuracy_score
from sklearn import neighbors
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

h = .02  # step size in the mesh

# Create color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])


for weights in ['uniform', 'distance']:
    n_neighbors = 15
    # Instance of Neighbours Classifier and fit the data.
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
    clf.fit(X_train,y_train)
    
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
#    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
#    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
#    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
#                         np.arange(y_min, y_max, h))
    preds_KNN = clf.predict(X_test)
    
    # Cross-validation Score
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(clf, x, y, cv=10, scoring='accuracy')
    print (scores)
 
    # Classification Report 
    from sklearn.metrics import classification_report

    print("KNN: \n" +classification_report(y_true=y_test, y_pred=preds_KNN))

    # Accuraccy_score
    print(accuracy_score(y_test, preds_KNN)*100)
    
    
    # Put the result into a color plot
#    Z = Z.reshape(xx.shape)
#    plt.figure()
#    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
#
#    # Plot also the training points
#    plt.scatter(x[:, 0], x[:, 1], c=y, cmap=cmap_bold,
#                edgecolor='k', s=20)
#    plt.xlim(xx.min(), xx.max())
#    plt.ylim(yy.min(), yy.max())
#    plt.title("3-Class classification (k = %i, weights = '%s')"
#              % (n_neighbors, weights))
#plt.show()
