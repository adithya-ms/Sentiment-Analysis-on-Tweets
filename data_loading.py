# -*- coding: utf-8 -*-

import pandas as pd

""" Loads in the twitter data"""
def load_data():
    tweets=pd.read_csv('data/text_emotion.csv')
    tweets.head()
    
    return tweets
