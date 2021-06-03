# -*- coding: utf-8 -*-
""" Contains code for the baseline models (kNN) and (SVM)"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

from sklearn.model_selection import train_test_split

# Functions partly from my ML code


""" Splits data into train and test set (for crossvalidation)"""
def splitData(inputs, labels):   
    # We split the dataset into 80% training data and 20% testing data
    inputs_train, inputs_test, labels_train, labels_test = train_test_split(
            inputs, labels, test_size=0.20) 
    
    return inputs_train, inputs_test, labels_train, labels_test


""" Implementation of the k-Nearest Neighbour classifier"""
def kNN(inputs_train, labels_train, inputs_test, k):    

    # Create a classifier with k neighbours
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(inputs_train, labels_train)
    
    # Predict the test data
    labels_prediction = classifier.predict(inputs_test)
    
    return labels_prediction


""" Implementation of the Support Vector Machine"""
def SVM(inputs_train, labels_train, inputs_test, mode):    
    
    # Create a classifier with kernel = mode
    classifier = svm.SVC(kernel=mode)    
    classifier.fit(inputs_train, labels_train)
    
    # Predict the test data
    labels_prediction = classifier.predict(inputs_test)

    return labels_prediction