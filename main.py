# -*- coding: utf-8 -*-

from data_loading import load_data
from baseline_models import kNN, SVM, print_confusion_matrix 
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
import statistics
#import bert_model
from transformers import BertTokenizer
from gensim.models import Word2Vec
from sklearn.dummy import DummyClassifier


""" Function for word2vec embeddings (unfortunately training on this took too long to be included) """ 
def word2vecEmbedding(contents):
    contents_split = []
    for line in contents:
        contents_split.append(line.split())

    # Train a word2vec model
    model_w2v = Word2Vec(sentences=contents_split, vector_size=10, window=5, min_count=1, workers=4)    
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


""" Returns wordpiece embeddings """ 
def wordpiece_tokens(contents, labels, feature_size):

	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
	wp_input = np.zeros((contents.shape[0],feature_size))
	wp_labels = []
	for ind in range(contents.shape[0]):
		text = contents.iloc[ind].lower()
		wp_labels.append(labels.iloc[ind].lower())
		encoded = tokenizer.encode(text, add_special_tokens=False)
		if len(encoded) < feature_size:
			while(len(encoded) < feature_size):
				encoded.append(0)
		wp_input[ind] = encoded
	
	le = LabelEncoder()
	wp_labels = le.fit_transform(wp_labels)
    
	return wp_input, wp_labels


""" Changes class labels from 13 -> 3 classes """ 
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

""" Function that uses k-fold cross validation to train models using different parameter settings """
def train(contents, labels, k, kernel, vectorizer):
    """ k-fold cross validation split on the train set"""
    no_folds = 5
    kf = KFold(n_splits=no_folds)	
    count = 1
    total_acc_knn = []
    total_acc_svm = []

    for train, test in kf.split(contents):
        print("Fold ", count)
        count += 1

        inputs_train, inputs_test, labels_train, labels_test = contents[train], contents[test], labels[train], labels[test]
    
        """ Vectorization """ 
        if vectorizer == "count":
            vectorizer = CountVectorizer(stop_words='english')
        elif vectorizer == "tfidf":
            vectorizer = TfidfVectorizer()
    
        inputs_train = vectorizer.fit_transform(inputs_train)
        inputs_test = vectorizer.transform(inputs_test)
    
        """ Baseline implementation"""
        predictionKNN = kNN(inputs_train, labels_train, inputs_test, k)
        predictionSVM = SVM(inputs_train, labels_train, inputs_test, kernel)
    
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
            
        acc_knn = correct_knn / total
        acc_svm = correct_svm / total
        print("accuracy KNN", acc_knn)
        print("accuracy SVM", acc_svm)
        total_acc_knn.append(acc_knn)
        total_acc_svm.append(acc_svm)
            
    print("Mean accuracy SVM:", np.mean(total_acc_svm))
    print("Standard deviation accuracy SVM:", statistics.stdev(total_acc_svm))


""" Function to do a final test on the held-out test set"""
def test(inputs_train, labels_train, inputs_test, labels_test, k, kernel, vectorizer):
    if vectorizer == 'count':
        vectorizer = CountVectorizer(stop_words='english')
    if vectorizer == 'tfidf':
        vectorizer = TfidfVectorizer()
    
    inputs_train = vectorizer.fit_transform(inputs_train)
    inputs_test = vectorizer.transform(inputs_test)      

    # Code to predict the majority class baseline (always predict the most frequent class)
    #dummy_clf = DummyClassifier(strategy = "most_frequent")
    #dummy_clf.fit(inputs_train, labels_train)
    #predictionDummy = dummy_clf.predict(inputs_test)
    
    print("Training the KNN and SVM..")
    # First train on all train data with optimal settings
    predictionSVM = SVM(inputs_train, labels_train, inputs_test, kernel)
    predictionKNN = kNN(inputs_train, labels_train, inputs_test, k)
        
    total = 0
    correct_svm = 0
    correct_knn = 0
    for i in range(len(labels_test)):
        if labels_test[i] == predictionKNN[i]:
            correct_knn += 1
        if labels_test[i] == predictionSVM[i]:
            correct_svm += 1
        total += 1   
    acc_svm = correct_svm / total
    acc_knn = correct_knn / total
    
    print("accuracy KNN", acc_knn) 
    print_confusion_matrix(labels_test, predictionKNN)   
    print("accuracy SVM", acc_svm)
    print_confusion_matrix(labels_test, predictionSVM)   
     
        
def main():
    data = load_data("data/train.csv")
    contents = data["content"]
    labels = data["sentiment"]
    labels_merged = []

    """ Change 13 -> 3 classes""" 
    labels_merged = changeLabelClasses(labels)
    
    """ Word2Vec embeddings (uncomment this and comment out the vectorizer to use it) """ 
    # This is not used in our final version of the code - training took too long
    #contents = word2vecEmbedding(contents)
	
    """ Grid search (hyperparameter tuning) for SVM and KNN"""
    #for kernel in ("linear", "poly", "rbf"):
    #    print("Kernel is ", kernel)
    #    train(contents, labels, 15, kernel, "count")
    #    print('\n')
    
    #for k in (13,15,17,19,21,23,25,27,29):
    #    print("k is ", k)
    #    train(contents, labels, k, "rbf", "count")
    #    print('\n')
    
    """ Load in test data for final experiments """    
    test_data = load_data("data/test.csv")
    test_contents = test_data["content"]
    test_labels = test_data["sentiment"]
    
    # Labels 13 -> 3 classes
    test_labels_merged = changeLabelClasses(test_labels) 
    
    """ Run the final experiments with the three different tokenizers"""
    for vect in ("count", "tfidf", "wordpiece"):
        print("Vectorizer is:", vect)
        # wordpiece embeddings
        if vect == "wordpiece":
            feature_size = 40
            contents, labels = wordpiece_tokens(contents,labels,feature_size)
            sc = StandardScaler()
            contents = sc.fit_transform(contents)           
            
        # 13 classes
        print("Results on 13 classes")
        test(contents, labels, test_contents, test_labels, 15, "rbf", vect)
        # 3 classes
        print("Results on 3 classes")
        test(contents, labels_merged, test_contents, test_labels_merged, 15, "rbf", vect)


if __name__ == "__main__":
	main()
