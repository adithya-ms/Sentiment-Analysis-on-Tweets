# -*- coding: utf-8 -*-

from data_loading import load_data
from baseline_models import splitData, kNN, SVM
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
import preprocessing
import pdb
import numpy as np

def main():
	data = load_data()
	#print(data)
	pdb.set_trace()
	contents = data["content"]
	labels = data["sentiment"]
	
    
  	# Test on a subset (with incomplete preprocessing -> large dictionary -> slow)
	#contents = contents[1:20000]
	#labels = labels[1:20000]
	
	""" Train - val - test Split """ # val + test is still to be implemented
	inputs_train, inputs_test, labels_train, labels_test = splitData(contents, labels)

	""" Preprocessing (not yet complete) """ 
	preprocessing.count_oov_words(inputs_train)
	inputs_train = preprocessing.remove_url(inputs_train)
	inputs_train = preprocessing.remove_mentions(inputs_train)
	inputs_train = preprocessing.remove_hashtags(inputs_train)
	inputs_train = preprocessing.remove_HTML(inputs_train)
	inputs_train = preprocessing.remove_grammar_abbreviations(inputs_train)
	inputs_train = preprocessing.remove_all_punctuation(inputs_train)
	inputs_train = preprocessing.remove_duplicate_spaces(inputs_train)
	#inputs_train = preprocessing.spell_checker(inputs_train)
	#inputs_train = preprocessing.alt_spell_checker(inputs_train)
	inputs_train = preprocessing.expand_acronym(inputs_train)
	preprocessing.count_oov_words(inputs_train)
	inputs_train = preprocessing.stemmer(inputs_train)
	inputs_train = preprocessing.remove_stopwords(inputs_train)

	'''
	pipe = Pipeline([('scaler', StandardScaler()),
					 ('svc', SVC())])
	'''

    
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


if __name__ == "__main__":
	main()
