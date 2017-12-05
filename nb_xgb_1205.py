#!/usr/bin/env python 
from __future__ import print_function 
from __future__ import division
from __future__ import absolute_import 

import sys
import re 
import os
import string
import random

import numpy as np
import pandas as pd 
import xgboost as xgb
from sklearn import ensemble, metrics, model_selection, naive_bayes 
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

import nltk
from nltk.corpus import stopwords

param = {'objective': 'binary:logistic',
        'eta': 0.001,
        'max_depth': 3, 
        'silent': 1, 
        'eval_metric': 'auc' 
        'min_child_weight': 'child',
        'subsample' 0.5, 
        'colsample_bytree': 'colsample',
        'seed': 2017
}

num_rounds = 1000

def main():
    #Set seed for reproducibility
    np.random.seed(0)
    
    print("Loading data...")
    #Load the data from the CSV files
    train_col=['company','label','text']
    train_file = "/data/scratch/lingrui/sec_temp/workspace/test_all.csv"
    training_data = load_data(train_file,train_col)
    print (training_data.shape[0])

def load_data(file_name,column_name):
    load_df = pd.DataFrame(columns=column_name)
    for pred_df in pd.read_csv(file_name,header=0,chunksize = 500):
        load_df = pd.concat([load_df,pred_df],axis=0)
    return load_df

def metaFeature(d):
    # Number of words in the text ##
    d["num_words"] = d["text"].apply(lambda x: len(str(x).split()))
    # Number of unique words in the text ##
    d["num_unique_words"] = d["text"].apply(lambda x: len(set(str(x).split())))
    # Number of characters in the text ##
    d["num_chars"] = d["text"].apply(lambda x: len(str(x)))
    # Number of stopwords in the text ##
    d["num_stopwords"] = d["text"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))
    # Number of punctuations in the text ##
    d["num_punctuations"] = d['text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )
    # Number of title case words in the text ##
    d["num_words_upper"] = d["text"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
    ## Number of title case words in the text ##
    d["num_words_title"] = d["text"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))
    ## Average length of the words in the text ##
    d["mean_word_len"] = d["text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))






if __name__ == '__main__':
    main()
