# -*- coding: utf-8 -*-

from data_loading import load_data
from baseline_models import splitData, kNN, SVM

data = load_data()
#print(data)

contents = data["content"]
labels = data["sentiment"]
print(labels)

""" Preprocessing (not yet implemented) """ 


""" Baseline implementation"""
inputs_train, inputs_test, labels_train, labels_test = splitData(contents, labels)

# Does not yet work because of missing preprocessing, so commented out
# predictionKNN = kNN(inputs_train, labels_train, inputs_test, 13)
# predictionSVM = SVM(inputs_train, labels_train, inputs_test, 'linear')

# Implement some accuracy calculation (predicted vs. true labels accuracy + analysis)