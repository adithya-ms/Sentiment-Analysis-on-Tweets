import pandas as pd

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





