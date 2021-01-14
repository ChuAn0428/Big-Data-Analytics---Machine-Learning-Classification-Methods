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

iris = load_iris() 
x, y = iris.data, iris.target
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=0)

# Decision Tree
dtree = tree.DecisionTreeClassifier().fit(x_train, y_train)
dtree_pred = dtree.predict(x_test)
con_dtree = confusion_matrix(dtree_pred, y_test)
acc_dtree = accuracy_score(y_test, dtree_pred)

# Naive Bayes
nb_gnb = GaussianNB()
nb_pred = nb_gnb.fit(x_train, y_train).predict(x_test)
con_nb = confusion_matrix(nb_pred, y_test)
acc_nb = accuracy_score(y_test, nb_pred)

# Logistic Regression
lr = LogisticRegression(random_state=0, max_iter=1000).fit(x_train, y_train)
lr_pred = lr.predict(x_test)
con_lr = confusion_matrix(lr_pred, y_test)
acc_lr = accuracy_score(y_test, lr_pred)

# KNN
knn = KNeighborsClassifier(n_neighbors=11).fit(x_train, y_train)
knn_pred = knn.predict(x_test)
con_knn = confusion_matrix(knn_pred, y_test)
acc_knn = accuracy_score(y_test, knn_pred)

def calculation(con_mat, y_true):
    # compute accuracy, precision, recall, and F-score
    # add up the predicted class 1,2,3 (row0,1,2)
    prow1 = con_mat[0][0] + con_mat[0][1] + con_mat[0][2] 
    prow2 = con_mat[1][0] + con_mat[1][1] + con_mat[1][2]
    prow3 = con_mat[2][0] + con_mat[2][1] + con_mat[2][2] 
    # add up the actual class 1,2,3 (column0,1,2)
    acol1 = con_mat[0][0] + con_mat[1][0] + con_mat[2][0] 
    acol2 = con_mat[0][1] + con_mat[1][1] + con_mat[2][1] 
    acol3 = con_mat[0][2] + con_mat[1][2] + con_mat[2][2] 

    #total = acol1 + acol2 + acol3
    # precision for each class and average
    prec1 = con_mat[0][0]/prow1  
    prec2 = con_mat[1][1]/prow2 
    prec3 = con_mat[2][2]/prow3 
    prec_average = (prec1 + prec2 + prec3)/float(len(con_mat))
    # the class-specific accuracy = precision
    acc1 = prec1 
    acc2 = prec2
    acc3 = prec3
    # overall accuracy
    acc_average = (con_mat[0][0]+con_mat[1][1]+con_mat[2][2])/float(len(y_true))

    # recall for each class and average
    recall1 = con_mat[0][0]/acol1
    recall2 = con_mat[1][1]/acol2
    recall3 = con_mat[2][2]/acol3
    recall_average = (recall1 + recall2 + recall3)/float(len(con_mat))

    # F-score for each class and average
    fscore1 = 2*con_mat[0][0]/(acol1+prow1)
    fscore2 = 2*con_mat[1][1]/(acol2+prow2)
    fscore3 = 2*con_mat[2][2]/(acol3+prow3)
    fscore_average = (fscore1 + fscore2 + fscore3)/float(len(con_mat))

    return round(acc1,3),round(prec1,3),round(recall1,3),round(fscore1,3),round(acc2,3),round(prec2,3),round(recall2,3),round(fscore2,3),round(acc3,3),round(prec3,3),round(recall3,3),round(fscore3,3),round(acc_average,3),round(prec_average,3),round(recall_average,3),round(fscore_average,3)

acc1,prec1,recall1,fscore1,acc2,prec2,recall2,fscore2,acc3,prec3,recall3,fscore3,acc_average,prec_average,recall_average,fscore_average = calculation(con_dtree, y_test)

print('\nIndicate class:')
print('Iris-setosa     -> 1')
print('Iris-versicolor -> 2')
print('Iris-virginica  -> 3')

print("\n1. Decision Trees:")
print("Confusion Matrix:")
print('               Actual')
print('               1  2  3')
print('predicted  1',con_dtree[0])
print('           2',con_dtree[1])
print('           3',con_dtree[2])

a = [acc1,prec1,recall1,fscore1]
b = [acc2,prec2,recall2,fscore2]
c = [acc3,prec3,recall3,fscore3]
d = [acc_average,prec_average,recall_average,fscore_average]

print('\nClassification Report:')
print('Class: accuracy | precision | recall | f1-score')
print('  1 :',a)
print('  2 :',b)
print('  3 :',c)
print(' Avg:',d)
print("Accuracy:", round(acc_dtree,3))

acc1,prec1,recall1,fscore1,acc2,prec2,recall2,fscore2,acc3,prec3,recall3,fscore3,acc_average,prec_average,recall_average,fscore_average = calculation(con_nb, y_test)

print("\n2. Naive Bayes:")
print("Confusion Matrix:")
print('               Actual')
print('               1  2  3')
print('predicted  1',con_nb[0])
print('           2',con_nb[1])
print('           3',con_nb[2])

a = [acc1,prec1,recall1,fscore1]
b = [acc2,prec2,recall2,fscore2]
c = [acc3,prec3,recall3,fscore3]
d = [acc_average,prec_average,recall_average,fscore_average]

print('\nClassification Report:')
print('Class: accuracy | precision | recall | f1-score')
print('  1 :',a)
print('  2 :',b)
print('  3 :',c)
print(' Avg:',d)
print("Accuracy:", round(acc_nb,3))

acc1,prec1,recall1,fscore1,acc2,prec2,recall2,fscore2,acc3,prec3,recall3,fscore3,acc_average,prec_average,recall_average,fscore_average = calculation(con_lr, y_test)

print("\n3. Logistic Regression:")
print("Confusion Matrix:")
print('               Actual')
print('               1  2  3')
print('predicted  1',con_lr[0])
print('           2',con_lr[1])
print('           3',con_lr[2])

a = [acc1,prec1,recall1,fscore1]
b = [acc2,prec2,recall2,fscore2]
c = [acc3,prec3,recall3,fscore3]
d = [acc_average,prec_average,recall_average,fscore_average]
print('\nClassification Report:')
print('Class: accuracy | precision | recall | f1-score')
print('  1 :',a)
print('  2 :',b)
print('  3 :',c)
print(' Avg:',d)
print("Accuracy:", round(acc_lr,3))

acc1,prec1,recall1,fscore1,acc2,prec2,recall2,fscore2,acc3,prec3,recall3,fscore3,acc_average,prec_average,recall_average,fscore_average = calculation(con_knn, y_test)

print("\n4. KNN:")
print("Confusion Matrix:")
print('               Actual')
print('               1  2  3')
print('predicted  1',con_knn[0])
print('           2',con_knn[1])
print('           3',con_knn[2])

a = [acc1,prec1,recall1,fscore1]
b = [acc2,prec2,recall2,fscore2]
c = [acc3,prec3,recall3,fscore3]
d = [acc_average,prec_average,recall_average,fscore_average]
print('\nClassification Report:')
print('Class: accuracy | precision | recall | f1-score')
print('  1 :',a)
print('  2 :',b)
print('  3 :',c)
print(' Avg:',d)
print("Accuracy:", round(acc_knn,3))
