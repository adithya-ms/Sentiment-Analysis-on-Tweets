# -*- coding: utf-8 -*-

from data_loading import load_data
from baseline_models import splitData, kNN, SVM
from sklearn.pipeline import Pipeline
import preprocessing
import pdb

def main():
	data = load_data()
	#print(data)

	contents = data["content"]
	labels = data["sentiment"]
	#print(labels)

	""" Train - val - test Split """
	inputs_train, inputs_test, labels_train, labels_test = splitData(contents, labels)

	""" Preprocessing (not yet implemented) """ 
	inputs_train = preprocessing.remove_url(inputs_train)
	inputs_train = preprocessing.remove_mentions(inputs_train)
	inputs_train = preprocessing.remove_hashtags(inputs_train)

	'''
	pipe = Pipeline([('scaler', StandardScaler()),
					 ('svc', SVC())])
	'''
	""" Baseline implementation"""

	# Does not yet work because of missing preprocessing, so commented out
	# predictionKNN = kNN(inputs_train, labels_train, inputs_test, 13)
	# predictionSVM = SVM(inputs_train, labels_train, inputs_test, 'linear')

	# Implement some accuracy calculation (predicted vs. true labels accuracy + analysis)




if __name__ == "__main__":
	main()
