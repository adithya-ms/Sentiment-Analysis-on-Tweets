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
	contents = data["content"]
	labels = data["sentiment"]
	
    
  	# Test on a subset (with incomplete preprocessing -> large dictionary -> slow)
	#contents = contents[1:20000]
	#labels = labels[1:20000]
	

	""" Preprocessing (not yet complete) """ 
	preprocessing.count_oov_words(contents)
	contents = preprocessing.remove_url(contents)
	contents = preprocessing.remove_mentions(contents)
	contents = preprocessing.remove_hashtags(contents)
	contents = preprocessing.remove_HTML(contents)
	contents = preprocessing.remove_grammar_abbreviations(contents)
	contents = preprocessing.remove_all_punctuation(contents)
	contents = preprocessing.remove_duplicate_spaces(contents)
	#contents = preprocessing.spell_checker(contents)
	#contents = preprocessing.alt_spell_checker(contents)
	contents = preprocessing.expand_acronym(contents)
	preprocessing.count_oov_words(contents)
	contents = preprocessing.stemmer(contents)
	contents = preprocessing.remove_stopwords(contents)

	'''
	pipe = Pipeline([('scaler', StandardScaler()),
					 ('svc', SVC())])
	'''

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


if __name__ == "__main__":
	main()
