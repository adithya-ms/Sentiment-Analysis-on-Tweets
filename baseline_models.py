# -*- coding: utf-8 -*-
""" Contains code for the baseline models (kNN) and (SVM)"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

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


""" Print a visualized confusion matrix"""
def print_confusion_matrix(y_true, y_pred):

   class_names = ["neutral", "worry", "happiness", "sadness", "love", "surprise", "fun", "relief", "hate", "empty", "enthusiasm", "boredom", "anger"]
   #class_names = ["pos","neg", "neutral"]
   fig, ax = plt.subplots(figsize=(15,15))

   cf_matrix = confusion_matrix(y_true, y_pred, labels = class_names)

   sns.set(font_scale=1.4)
   res = sns.heatmap(cf_matrix, annot=True, xticklabels = class_names, yticklabels = class_names, ax=ax, annot_kws={"size": 22}, fmt='g')
   res.set_xticklabels(res.get_xmajorticklabels(), fontsize = 25, rotation=45)
   res.set_yticklabels(res.get_ymajorticklabels(), fontsize = 25, rotation=0)
   plt.title('Confusion matrix - true (vertical) vs. predicted (horizontal) labels', fontsize = 28)
   plt.savefig('conf_mat.png')