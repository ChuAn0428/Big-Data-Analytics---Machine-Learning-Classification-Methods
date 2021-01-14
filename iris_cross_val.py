# -*- coding: utf-8 -*-
#################################
# Author: Chu-An Tsai
# 2/23/2020
#################################

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report

iris = load_iris() 
x, y = iris.data, iris.target


# Decision Tree
dtree = tree.DecisionTreeClassifier()
scores_dt = cross_val_score(dtree, x, y, cv=5)
print("Dataset: iris.data")
print("3-fold cross validation")
print("\n1. Decision tree")
for i in range(1,len(scores_dt)+1):
    print("   accuracy",i,"=",round(scores_dt[i-1],3))
print("   Average accuracy:",round(sum(scores_dt)/len(scores_dt),3))

# Naive Bayes
nb_gnb = GaussianNB()
scores_nb = cross_val_score(nb_gnb, x, y, cv=5)
print("\n2. Naive Bayes")
for i in range(1,len(scores_nb)+1):
    print("   accuracy",i,"=",round(scores_nb[i-1],3))
print("   Average accuracy:",round(sum(scores_nb)/len(scores_nb),3))

# Logistic Regression
lr = LogisticRegression(random_state=0, max_iter=1000)
scores_lr = cross_val_score(lr, x, y, cv=5)
print("\n3. Logistic Regression")
for i in range(1,len(scores_lr)+1):
    print("   accuracy",i,"=",round(scores_lr[i-1],3))
print("   Average accuracy:",round(sum(scores_lr)/len(scores_lr),3))

# KNN
knn = KNeighborsClassifier(n_neighbors=7)
scores_knn = cross_val_score(knn, x, y, cv=5)
print("\n4. k-Nearest Neighbor")
for i in range(1,len(scores_knn)+1):
    print("   accuracy",i,"=",round(scores_knn[i-1],3))
print("   Average accuracy:",round(sum(scores_knn)/len(scores_knn),3))
