# -*- coding: utf-8 -*-

from data_loading import load_data
from baseline_models import splitData, kNN, SVM, print_confusion_matrix 
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
#import preprocessing
import pdb
import numpy as np
import statistics
#import bert_model

from gensim.models import Word2Vec

def word2vecEmbedding(contents):
    contents_split = []
    for line in contents:
        contents_split.append(line.split())

    # Train a word2vec model
    model_w2v = Word2Vec(sentences=contents_split, vector_size=10, window=5, min_count=1, workers=4)
    # print(model_w2v.wv.most_similar('man'))
    
    max_length = max(map(len, contents_split))

    embeddings_input = []
    for sentence in contents_split:
        sentence_embedding = []
        for word in sentence:
            sentence_embedding.append(np.array(model_w2v.wv[word]))
        while(len(sentence_embedding) < max_length):
            sentence_embedding.append(np.zeros(10))
        embeddings_input.append(sentence_embedding)

    embeddings_input = np.array(embeddings_input)
    
    dim1, dim2, dim3 = embeddings_input.shape
    embeddings_input = embeddings_input.reshape((dim1, dim2 * dim3))
    print(embeddings_input.shape)
    
    return embeddings_input

def changeLabelClasses(labels):
	neutral = []
	pos = []
	neg = []
	labels_merged = []

	for label in labels:
		if label == "anger":
			labels_merged.append("neg")
		elif label == "boredom":
			labels_merged.append("neg")
		elif label == "hate":
			labels_merged.append("neg")
		elif label == "sadness":
			labels_merged.append("neg")
		elif label == "worry":
			labels_merged.append("neg")
		elif label == "empty":
			labels_merged.append("neutral")
		elif label == "enthusiasm":
			labels_merged.append("pos")
		elif label == "fun":
			labels_merged.append("pos")
		elif label == "happiness":
			labels_merged.append("pos")
		elif label == "love":
			labels_merged.append("pos")
		elif label == "relief":
			labels_merged.append("pos")
		elif label == "surprise":
			labels_merged.append("neutral")
		else:
			labels_merged.append("neutral")
	for label in labels_merged:
		if label == "neutral":
			neutral.append(label)
		elif label == "pos":
			pos.append(label)
		elif label == "neg":
			neg.append(label)
	print("label dist:")
	print("pos= ", len(pos))
	print("neg= ", len(neg))
	print("neutral= ", len(neutral))
	return labels_merged

def main():
    data = load_data("data/train.csv")
    contents = data["content"]
    labels = data["sentiment"]
    labels_merged = []
    no_folds = 5

    """ Change 13 -> 3 classes""" 
    labels_merged = changeLabelClasses(labels)
    
    """ Word2Vec embeddings (uncomment this and comment out the vectorizer to use it) """ 
    #contents = word2vecEmbedding(contents)

    """ k-fold cross validation split on the train set"""
    kf = KFold(n_splits=no_folds)	
    count = 1
    total_acc = []
    best_acc = 0
    for train, test in kf.split(contents):
        print("Fold ", count)
        count += 1
        inputs_train, inputs_test, labels_train, labels_test = contents[train], contents[test], labels[train], labels[test]
        #inputs_train, inputs_test, labels_train, labels_test = splitData(contents, labels)
    
        """ Vectorization (first version) """ 
        vectorizer = CountVectorizer(stop_words='english')
        #vectorizer = TfidfVectorizer()
        #vectorizer = HashingVectorizer()
    
        inputs_train = vectorizer.fit_transform(inputs_train)
        inputs_test = vectorizer.transform(inputs_test)
    
    
        """ Baseline implementation"""
        predictionKNN = kNN(inputs_train, labels_train, inputs_test, 13)
        #predictionSVM = SVM(inputs_train, labels_train, inputs_test, 'linear')
    
        """ Performance calculation"""
        labels_test = np.array(labels_test)
        correct_knn = 0
        #correct_svm = 0
        total = 0
    
        for i in range(len(labels_test)):
            if labels_test[i] == predictionKNN[i]:
                correct_knn += 1
            #if labels_test[i] == predictionSVM[i]:
            #    correct_svm += 1
            total += 1
            
        acc = correct_knn / total
        total_acc.append(acc)
        print("accuracy KNN", acc)
        if acc > best_acc:
            best_prediction = predictionKNN
            best_labels = labels_test
        #print("Confusion matrix KNN:")
        #print_confusion_matrix(labels_test, predictionKNN)
        #print("accuracy SVM", correct_svm / total)
        #print("Confusion matrix SVM:")
        #print_confusion_matrix(labels_test, predictionSVM)
    
    #bert_model.bert_ops(inputs_train, inputs_test, labels_train, labels_test, batch_size = 32, epochs = 5)
    print("Mean accuracy KNN:", np.mean(total_acc))
    print("Standard deviation accuracy KNN:", statistics.stdev(total_acc))
    # Print the confusion matrix of the best fold
    print_confusion_matrix(best_labels, best_prediction)

if __name__ == "__main__":
	main()
