# -*- coding: utf-8 -*-

from data_loading import load_data
from baseline_models import splitData, kNN, SVM
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
import preprocessing
import pdb
import numpy as np
import bert_model
def main():
	
	data = load_data("data/preprocessed_data.csv")
	contents = data["content"]
	labels = data["sentiment"]

	""" Train - val - test Split """ # val + test is still to be implemented
	inputs_train, inputs_test, labels_train, labels_test = splitData(contents, labels)
    
	""" Vectorization (first version) """ 
	vectorizer = CountVectorizer(stop_words='english')
	inputs_train = vectorizer.fit_transform(inputs_train)
	inputs_test = vectorizer.transform(inputs_test)
    
	""" Baseline implementation"""

	predictionKNN = kNN(inputs_train, labels_train, inputs_test, 13)
	predictionSVM = SVM(inputs_train, labels_train, inputs_test, 'linear')

	""" Performance calculation"""
	labels_test = np.array(labels_test)
	correct_knn = 0
	correct_svm = 0
	total = 0

	for i in range(len(labels_test)):
		if labels_test[i] == predictionKNN[i]:
			correct_knn += 1
		if labels_test[i] == predictionSVM[i]:
			correct_svm += 1
		total += 1

	print("accuracy KNN", correct_knn / total)
	print("accuracy SVM", correct_svm / total)

	#bert_model.bert_ops(inputs_train, inputs_test, labels_train, labels_test, batch_size = 32, epochs = 5)

if __name__ == "__main__":
	main()
