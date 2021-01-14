# -*- coding: utf-8 -*-
#################################
# Author: Chu-An Tsai
# 2/23/2020
#################################


from sklearn.datasets import load_iris
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

dataset = np.loadtxt("house-votes-84.data", delimiter=',', dtype=str)
newdataset = dataset.copy()

for i in range(len(newdataset)):
    for j in range(1, len(newdataset.T)):
        if (newdataset[i][j] == 'y'):
            newdataset[i][j] = '1'
        elif (newdataset[i][j] == 'n'):
            newdataset[i][j] = '-1'
        else: 
            newdataset[i][j] = '0'

    if newdataset[i][0] == 'republican':
        newdataset[i][0] = 1 
    else:
        newdataset[i][0] = 2
newdataset = newdataset.astype(int)      
  
data = newdataset[:,1:17].copy()
data_label = newdataset[:,0].copy()

print("Dataset: house-votes-84.data")

##### 1. Linear Kernel SVM
print('\n1. Linear Kernel:')

# 1-1. Training set = first 75% data, Tuning set = 25% from training set, Test set = last 25% data 

data_train_lin_1, data_test_lin_1, data_train_label_lin_1, data_test_label_lin_1 = train_test_split(data, data_label, train_size=0.75, random_state = 0, stratify = data_label)

#C = np.arange(0.1, 5, 0.02)
#parameters_linear = [{'C':C}]
parameters_linear = [{'C':[ 0.01, 0.05, 0.1, 0.5, 1.0, 2, 5, 10]}]

model_linear = GridSearchCV(SVC(kernel='linear'), parameters_linear, cv=3).fit(data_train_lin_1, data_train_label_lin_1)
print('The best parameters: ', model_linear.best_params_)
#print("Scores for crossvalidation:")
#for mean, params in zip(model_linear.cv_results_['mean_test_score'], model_linear.cv_results_['params']):
    #print("Accuracy: %0.6f for %r" % (mean, params))
predicted_label_lin_1 =  model_linear.predict(data_test_lin_1)
accuracy_lin_1 = accuracy_score(data_test_label_lin_1, predicted_label_lin_1)
print('accuracy:',round(accuracy_lin_1,3))

# 1-2. Training set = last 75% data, Tuning set = 25% from training set, Test set = first 25% data 

data_test_lin_2, data_train_lin_2, data_test_label_lin_2, data_train_label_lin_2 = train_test_split(data, data_label, train_size=0.25, random_state = 0, stratify = data_label)

model_linear = GridSearchCV(SVC(kernel='linear'), parameters_linear, cv=3).fit(data_train_lin_2, data_train_label_lin_2)
print('The best parameters: ', model_linear.best_params_)
#print("Scores for crossvalidation:")
#for mean, params in zip(model_linear.cv_results_['mean_test_score'], model_linear.cv_results_['params']):
    #print("Accuracy: %0.6f for %r" % (mean, params))
predicted_label_lin_2 =  model_linear.predict(data_test_lin_2)
accuracy_lin_2 = accuracy_score(data_test_label_lin_2, predicted_label_lin_2)
print('accuracy:',round(accuracy_lin_2,3))

# 1-3. Training set = first 37.5% and last 37.5%, Tuning set = 25% from training set, Test set = first 25% data 

data_temp1_lin_3, data_temp2_lin_3, data_temp1_label_lin_3, data_temp2_label_lin_3 = train_test_split(data, data_label, train_size=0.375, random_state = 0, stratify = data_label)
data_test_lin_3, data_temp3_lin_3, data_test_label_lin_3, data_temp3_label_lin_3 = train_test_split(data_temp2_lin_3, data_temp2_label_lin_3, train_size=0.4, random_state = 0, stratify = data_temp2_label_lin_3)
data_train_lin_3 = np.vstack((data_temp1_lin_3, data_temp3_lin_3))
data_train_label_lin_3 = np.hstack((data_temp1_label_lin_3, data_temp3_label_lin_3))

model_linear = GridSearchCV(SVC(kernel='linear'), parameters_linear, cv=3).fit(data_train_lin_3, data_train_label_lin_3)
print('The best parameters: ', model_linear.best_params_)
#print("Scores for crossvalidation:")
#for mean, params in zip(model_linear.cv_results_['mean_test_score'], model_linear.cv_results_['params']):
    #print("Accuracy: %0.6f for %r" % (mean, params))
predicted_label_lin_3 =  model_linear.predict(data_test_lin_3)
accuracy_lin_3 = accuracy_score(data_test_label_lin_3, predicted_label_lin_3)
print('accuracy:',round(accuracy_lin_3,3))

scores_lin = np.array([accuracy_lin_1, accuracy_lin_2, accuracy_lin_3])
print('Average 3-fold accuracy (with std):', round(scores_lin.mean(),3), '(+/-',round(scores_lin.std(),3),')')


##### 2. Gaussion Kernel SVM (RBF)
print('\n2. Gaussion Kernel (rbf):')

# 2-1. Training set = first 75% data, Tuning set = 25% from training set, Test set = last 25% data 

data_train_rbf_1, data_test_rbf_1, data_train_label_rbf_1, data_test_label_rbf_1 = train_test_split(data, data_label, train_size=0.75, random_state = 0, stratify = data_label)

#C = np.arange(0.1, 5, 0.05)
#G = np.arange(0.01, 0.5, 0.01)
#parameters_rbf = [{'C': C, 'gamma': G}]
parameters_rbf = [{'C': [0.01, 0.05, 0.1, 0.5, 1.0, 2, 5, 10], 'gamma': [0.01, 0.05, 0.1]}]

model_rbf = GridSearchCV(SVC(kernel='rbf'), parameters_rbf, cv=3).fit(data_train_rbf_1, data_train_label_rbf_1)
print('The best parameters: ', model_rbf.best_params_)
#print("Scores for crossvalidation:")
#for mean, params in zip(model_rbf.cv_results_['mean_test_score'], model_rbf.cv_results_['params']):
    #print("Accuracy: %0.6f for %r" % (mean, params))
predicted_label_rbf_1 =  model_rbf.predict(data_test_rbf_1)
accuracy_rbf_1 = accuracy_score(data_test_label_rbf_1, predicted_label_rbf_1)
print('accuracy:',round(accuracy_rbf_1,3))

# 2-2. Training set = last 75% data, Tuning set = 25% from training set, Test set = first 25% data 

data_test_rbf_2, data_train_rbf_2, data_test_label_rbf_2, data_train_label_rbf_2 = train_test_split(data, data_label, train_size=0.25, random_state = 0, stratify = data_label)

model_rbf = GridSearchCV(SVC(kernel='rbf'), parameters_rbf, cv=3).fit(data_train_rbf_2, data_train_label_rbf_2)
print('The best parameters: ', model_rbf.best_params_)
#print("Scores for crossvalidation:")
#for mean, params in zip(model_rbf.cv_results_['mean_test_score'], model_rbf.cv_results_['params']):
    #print("Accuracy: %0.6f for %r" % (mean, params))
predicted_label_rbf_2 =  model_rbf.predict(data_test_rbf_2)
accuracy_rbf_2 = accuracy_score(data_test_label_rbf_2, predicted_label_rbf_2)
print('accuracy:',round(accuracy_rbf_2,3))

# 2-3. Training set = first 37.5% and last 37.5%, Tuning set = 25% from training set, Test set = first 25% data 

data_temp1_rbf_3, data_temp2_rbf_3, data_temp1_label_rbf_3, data_temp2_label_rbf_3 = train_test_split(data, data_label, train_size=0.375, random_state = 0, stratify = data_label)
data_test_rbf_3, data_temp3_rbf_3, data_test_label_rbf_3, data_temp3_label_rbf_3 = train_test_split(data_temp2_rbf_3, data_temp2_label_rbf_3, train_size=0.4, random_state = 0, stratify = data_temp2_label_rbf_3)
data_train_rbf_3 = np.vstack((data_temp1_rbf_3, data_temp3_rbf_3))
data_train_label_rbf_3 = np.hstack((data_temp1_label_rbf_3, data_temp3_label_rbf_3))

model_rbf = GridSearchCV(SVC(kernel='rbf'), parameters_rbf, cv=3).fit(data_train_rbf_3, data_train_label_rbf_3)
print('The best parameters: ', model_rbf.best_params_)
#print("Scores for crossvalidation:")
#for mean, params in zip(model_rbf.cv_results_['mean_test_score'], model_rbf.cv_results_['params']):
    #print("Accuracy: %0.6f for %r" % (mean, params))
predicted_label_rbf_3 =  model_rbf.predict(data_test_rbf_3)
accuracy_rbf_3 = accuracy_score(data_test_label_rbf_3, predicted_label_rbf_3)
print('accuracy:',round(accuracy_rbf_3,3))

scores_rbf = np.array([accuracy_rbf_1, accuracy_rbf_2, accuracy_rbf_3])
print('Average 3-fold accuracy (with std):', round(scores_rbf.mean(),3), '(+/-', round(scores_rbf.std(),3),')')

