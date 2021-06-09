# -*- coding: utf-8 -*-

import pandas as pd

""" Loads in the twitter data"""
def load_data(file_path):
    tweets=pd.read_csv(file_path)
    tweets.head()
    
    return tweets
