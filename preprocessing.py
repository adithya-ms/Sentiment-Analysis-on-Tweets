import pandas as pd
import re
from spellchecker import SpellChecker
import enchant
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

def remove_HTML(data):
    for index in range(len(data)):
        HTML = ["&amp;", " and ", "&quot;", "", "&lt;", "", "&gt;", ""]
        for i in range(len(HTML)-1):
            if(i%2 == 0):
                data.iloc[index] = data.iloc[index].replace(HTML[i], HTML[i+1])
        
    return data  

def remove_grammar_abbreviations(data):
    for index in range(len(data)):
        Abr = ["'m", " am", "'re", " are", "'ve", " have", "'ll", " will", "n't", " not", "'d", " had", "'s", " is"]
        for i in range(len(Abr)-1):
            if(i%2 == 0):
                data.iloc[index] = data.iloc[index].replace(Abr[i], Abr[i+1])
        
    return data 

def remove_url(data):
	for index in range(len(data)):
		tweet_list = data.iloc[index].split(" ")
		for word in tweet_list:
			if "http://" in word or "https://" in word:
				data.iloc[index] = data.iloc[index].replace(word, "")
			else:
				pass
	return data


def remove_mentions(data):
	for index in range(len(data)):
		tweet_list = data.iloc[index].split(" ")
		for word in tweet_list:
			if len(word) < 2:
				continue
			elif word[0] == "@":
				data.iloc[index] = data.iloc[index].replace(word, word[1:])
			else:
				pass
	return data

def remove_hashtags(data):
	for index in range(len(data)):
		tweet_list = data.iloc[index].split(" ")
		for word in tweet_list:
			if len(word) < 2:
				continue
			elif word[0] == "#":
				data.iloc[index] = data.iloc[index].replace(word, word[1:])
			else:
				pass
	return data

def remove_all_punctuation(data):
    for index in range(len(data)):
        data.iloc[index] = re.sub(r'[^\w\s]', ' ', data.iloc[index])
    return data

def remove_duplicate_spaces(data):
    for index in range(len(data)):
        data.iloc[index] = re.sub(r'\s+', " ", data.iloc[index])
        data.iloc[index] = data.iloc[index].strip()
    return data
	
def spell_checker(data):
    spell = SpellChecker()
    for index in range(len(data)):
        tweet_list = data.iloc[index].split(" ")
        new_sentence = []
        for word in tweet_list:
            if str(word) in spell:
                new_sentence.append(word)
            else:
                new_sentence.append(spell.correction(str(word)))
        new_sentence = " ".join(new_sentence)
        if data.iloc[index] != new_sentence:
            data.iloc[index] = new_sentence
    return data

def enchant_spell_checker(data):
    en = enchant.Dict("en_US")
    for index in range(len(data)):
        tweet_list = data.iloc[index].split(" ")
        new_sentence = []
        for word in tweet_list:
            if en.check(str(word)):
                new_sentence.append(word)
            else:
                new_sentence.append(en.suggest(str(word))[0])
        new_sentence = " ".join(new_sentence)
        if data.iloc[index] != new_sentence:
            data.iloc[index] = new_sentence
    return data

def stemmer(data):
    ps = PorterStemmer()
    for index in range(len(data)):
        tweet_list = data.iloc[index].split(" ")
        new_sentence = []
        for word in tweet_list:
            new_sentence.append(ps.stem(word))
        new_sentence = " ".join(new_sentence)
        data.iloc[index] = new_sentence
    return data  

def remove_stopwords(data):
    for index in range(len(data)):
        tokens = data.iloc[index].split(" ")
        filtered_text = [t for t in tokens if not t in stopwords.words("english")]
        data.iloc[index] = " ".join(filtered_text)
    return data