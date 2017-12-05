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

def metaFeature(df):
    ## Number of words in the text ##
if __name__ == '__main__':
    main()
