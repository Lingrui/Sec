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
        'eval_metric': 'auc', 
        'min_child_weight': 'child',
        'subsample': 0.5, 
        'colsample_bytree': 'colsample',
        'seed': 2017
}

num_rounds = 1000

train_col=['company','label','text']
train_file = "/data/scratch/lingrui/sec_temp/workspace/test_all.csv"
test_col=['company','label','text']
test_file = "/data/scratch/lingrui/sec_temp/workspace/test_all.csv"

def main():
    #Set seed for reproducibility
    np.random.seed(0)
    os.system('date')
    print("Loading data...")
    #Load the data from the CSV files
    training_data = load_data(train_file,train_col)
    test_data = load_data(test_file,test_col)
    print ("TFIDF Vectorize data...")
    os.system('date')

    x,y,z = TfidfV(training_data,test_data,'char')
    print (x.shape)

    print ('Count Vectorize data...')
    
    os.system('date')
    os.system('date')
    os.system('date')
    os.system('date')

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

### Fit transform the tfidf vectorizer ###
def TfidfV(train_df,test_df,split_type):
    tfidf_vec = TfidfVectorizer(stop_words='english', ngram_range=(1,1),analyzer=split_type)
    full_tfidf = tfidf_vec.fit_transform(train_df['text'].values.tolist() + test_df['text'].values.tolist())
    train_tfidf = tfidf_vec.transform(train_df['text'].values.tolist())
    test_tfidf = tfidf_vec.transform(test_df['text'].values.tolist())
    return full_tfidf, train_tfidf, test_tfidf 

### Fit transform the count vectorizer ###
def CountV(train_df,test_df,split_type):
    count_vec = CountVectorizer(stop_words='english', ngram_range=(1,1),analyzer=split_type)
    count_vec.fit(train_df['text'].values.tolist() + test_df['text'].values.tolist())
    train_count = count_vec.transform(train_df['text'].values.tolist())
    test_count = count_vec.transform(test_df['text'].values.tolist())
    return train_count,test_count

### SVD on TFIDF 
def SVD(train_df,test_df,split_type,full_tfidf,train_tfidf,test_tfidf):
    n_comp = 20  #value for LSA
    svd_obj = TruncatedSVD(n_components=n_comp, algorithm='arpack')
    svd_obj.fit(full_tfidf)
    train_svd = pd.DataFrame(svd_obj.transform(train_tfidf))
    test_svd = pd.DataFrame(svd_obj.transform(test_tfidf))
        
    train_svd.columns = ['svd_'+split_type+'_'+str(i) for i in range(n_comp)]
    test_svd.columns = ['svd_'+split_type+'_'+str(i) for i in range(n_comp)]
    train_df = pd.concat([train_df, train_svd], axis=1)
    test_df = pd.concat([test_df, test_svd], axis=1)
    return train_df,test_df

def runMNB(train_X,train_y,test_X):
    model = naive_bayes.MultinomialNB()
    model.fit(train_X,train_y)
    pred_test_y = model.predict_proba(test_X)




if __name__ == '__main__':
    main()
