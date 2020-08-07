# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 11:06:21 2020

@author: Gurudas
"""

#importing relevant libraries
import numpy as np
import warnings
from sklearn.datasets import load_breast_cancer
warnings.filterwarnings('ignore')

#loading in the dataset 
cancer = load_breast_cancer()

#checking out the feature names and the taget variable names 
print("Features: ",cancer.feature_names,'\n')
print("Labels: ",cancer.target_names)

print(cancer.data.shape)
##### The above results shows that it has 30 features and  569 observations

## The dataset have 2 target values i.e. malignant and benign
print(cancer.target_names)

## Defining variables 
X = np.array(cancer.data)
y = np.array(cancer.target)

##### Splitting the data 

from sklearn.model_selection import train_test_split

# 70% training and 30% test
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3,random_state=0) 

###### Fitting the model and training the dataset with different parameter values  
#(HyperParameter Tuning)

from sklearn.svm import SVC
from sklearn.metrics import classification_report

def model_training(kernelInput,gammaInput,C_input):
    print("\n\n\n###########  MODEL PARAMETERS  #############\n\n\n")
    print("*****************SVC with kernel as {},gamma as {} and C as {}**********************".format(kernelInput,gammaInput,C_input))
    
    #creating object for model
    classifier = SVC(kernel=kernelInput,gamma=gammaInput,C=C_input)
    
    print(classifier)
    
    #fitting the model to the data
    classifier.fit(X_train,y_train)
    
    #predicting the test data
    prediction = classifier.predict(X_test)

    print("\n\n\n###########  MODEL EVALUATION  #############\n")
    print("************Classification Report****************")
    print(classification_report(y_test, prediction))
    
model_training('linear', 0.1, 0.1)
model_training('rbf', 0.1, 0.1)
model_training('sigmoid', 0.1, 0.1)


# =============================================================================
# Finding out the best parameters for getting the good accuracy for the model
# =============================================================================

from sklearn.model_selection import GridSearchCV

def svm_parameter_selection():
    #defining parameter range
    param_grid = { 'C' :[0.01,0.1,1,10,100],
                   'gamma' : [0.1,0.01,0.001,0.0001],
                   'kernel': ['rbf','linear','sigmoid']
                 }
    
    #creating model
    print("***** Displaying the model parameters\n")
    grid_search = GridSearchCV(SVC(), param_grid,verbose=3)
    
    #fitting the data
    grid_search.fit(X_train,y_train)
    
    print("\n\n\n*********Displaying the best parameters after tuning*********\n")
    print(grid_search.best_params_)
    
    print("\n************* Model after hyperparameter tuning**************")
    print(grid_search.best_estimator_)
    
    print("\n\n**************** Evaluation metrics  after hyper parameter tuning*************\n")
    pred = grid_search.predict(X_test)
    print(classification_report(y_test, pred))
    

svm_parameter_selection()


# =============================================================================
# Output : 
# =============================================================================


###########  MODEL PARAMETERS  #############



*****************SVC with kernel as linear,gamma as 0.1 and C as 0.1**********************
SVC(C=0.1, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma=0.1, kernel='linear',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)



###########  MODEL EVALUATION  #############

************Classification Report****************
              precision    recall  f1-score   support

           0       0.88      0.97      0.92        63
           1       0.98      0.93      0.95       108

    accuracy                           0.94       171
   macro avg       0.93      0.95      0.94       171
weighted avg       0.94      0.94      0.94       171




###########  MODEL PARAMETERS  #############



*****************SVC with kernel as rbf,gamma as 0.1 and C as 0.1**********************
SVC(C=0.1, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma=0.1, kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)



###########  MODEL EVALUATION  #############

************Classification Report****************
              precision    recall  f1-score   support

           0       0.00      0.00      0.00        63
           1       0.63      1.00      0.77       108

    accuracy                           0.63       171
   macro avg       0.32      0.50      0.39       171
weighted avg       0.40      0.63      0.49       171




###########  MODEL PARAMETERS  #############



*****************SVC with kernel as sigmoid,gamma as 0.1 and C as 0.1**********************
SVC(C=0.1, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma=0.1, kernel='sigmoid',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)



###########  MODEL EVALUATION  #############

************Classification Report****************
              precision    recall  f1-score   support

           0       0.00      0.00      0.00        63
           1       0.63      1.00      0.77       108

    accuracy                           0.63       171
   macro avg       0.32      0.50      0.39       171
weighted avg       0.40      0.63      0.49       171




########################## Using GridSearchCV   ###############################


*********Displaying the best parameters after tuning*********

{'C': 0.1, 'gamma': 0.1, 'kernel': 'linear'}

************* Model after hyperparameter tuning**************
SVC(C=0.1, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma=0.1, kernel='linear',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)


**************** Evaluation metrics  after hyper parameter tuning*************

              precision    recall  f1-score   support

           0       0.88      0.97      0.92        63
           1       0.98      0.93      0.95       108

    accuracy                           0.94       171
   macro avg       0.93      0.95      0.94       171
weighted avg       0.94      0.94      0.94       171

[Parallel(n_jobs=1)]: Done 300 out of 300 | elapsed:  3.0min finished

    
    
    



    
    
    
    
    
    
    
    
