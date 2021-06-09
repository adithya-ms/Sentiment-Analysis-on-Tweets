import pandas as pd
import re
from spellchecker import SpellChecker
import enchant
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pdb
import json
import string
from gingerit.gingerit import GingerIt

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

def change_emoticons(data):
    emote=pd.read_csv('emoticons_lookup_table.csv', sep = ';')
    emoticons = emote['emoticon']
    words = emote['word']
    for index in range(len(data)):
        tweet_list = data.iloc[index].split(" ")
        for word in tweet_list:
            for i in range(len(emoticons)):
                data.iloc[index] = data.iloc[index].replace(emoticons[i], words[i])
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
				data.iloc[index] = data.iloc[index].replace(word, "")
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
        data.iloc[index] = re.sub(r'[^0-9a-zA-Z\s]', ' ', data.iloc[index])
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


def expand_acronym(data):
    slang_dict = pd.read_csv('data/Slang_lookup_table.csv')
    eng_dict = enchant.Dict("en_US")
    for index in range(len(data)):
        tweet_list = data.iloc[index].split(" ")
        for word in tweet_list:
            if len(word) > 1:
                if eng_dict.check(word) is False:
                    if word.upper() in list(slang_dict["Word"]):
                        if len(slang_dict.loc[slang_dict["Word"] == word]["Meaning"]) > 0:
                            abbr = slang_dict.loc[slang_dict["Word"] == word]["Meaning"].values[0]
                            data.iloc[index] = data.iloc[index].replace(word, abbr.lower())
                        else:
                            pass
                    else:
                        pass
                else:
                    pass
    return data
            
                
def alt_spell_checker(data):
    parser = GingerIt()
    for index in range(len(data)):
        corrected_tweet = parser.parse(data.loc[index])
        data.loc[index] = corrected_tweet['result'].lower()
    return data

def count_oov_words(data):
    spell = SpellChecker()
    iv = []
    oov = []
    for index in range(len(data)):
        tweet_list = data.iloc[index].split(" ")
        new_sentence = []
        for word in tweet_list:
            if str(word) in spell and not(word in iv):
                iv.append(word)
            elif not(str(word) in spell) and not(word in oov):
                oov.append(word)

    print("Unique in vocabulary word count:", str(len(iv)))
    print("Unique out of vocabulary word count:", str(len(oov)))
    
