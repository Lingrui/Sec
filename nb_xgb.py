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
from sklearn.model_selection import KFold

import nltk
from nltk.corpus import stopwords

param = {'objective': 'binary:logistic',
        'n_estimators':1000,
        'learning_rate': 0.001,
        'max_depth': 3, 
        'silent': 1, 
#'min_child_weight': 'child',
        'subsample': 0.5, 
        'colsample_bytree': 0.60,
        'seed': 2017,
        'reg_lambda': 1,
        'reg_alpha': 1
}


eng_stopwords = set(stopwords.words("english"))
pd.options.mode.chained_assignment = None
<<<<<<< HEAD
## Read the train and test dataset and check the top few lines ##

train_df = pd.read_csv("/home/lcai/s2/sec_10k/workspace/train.csv",header=0)
test_df = pd.read_csv("/home/lcai/s2/sec_10k/workspace/test.csv",header=0)
train_y = train_df['label'].values
train_id = train_df['company'].values
tet_id = test_df['company'].values

print("Number of rows in dataset : ", train_df.shape[0])
##################################################

##META features###
def metaFeature(train_df,test_df):
	## Number of words in the text ##
	train_df["num_words"] = train_df["text"].apply(lambda x: len(str(x).split()))
	test_df["num_words"] = test_df["text"].apply(lambda x: len(str(x).split()))
	## Number of unique words in the text ##
	train_df["num_unique_words"] = train_df["text"].apply(lambda x: len(set(str(x).split())))
	test_df["num_unique_words"] = test_df["text"].apply(lambda x: len(set(str(x).split())))
	## Number of characters in the text ##
	train_df["num_chars"] = train_df["text"].apply(lambda x: len(str(x)))
	test_df["num_chars"] = test_df["text"].apply(lambda x: len(str(x)))
	## Number of stopwords in the text ##
	train_df["num_stopwords"] = train_df["text"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))
	test_df["num_stopwords"] = test_df["text"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))
	## Number of punctuations in the text ##
	train_df["num_punctuations"] =train_df['text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )
	test_df["num_punctuations"] =test_df['text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )
	## Number of title case words in the text ##
	train_df["num_words_upper"] = train_df["text"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
	test_df["num_words_upper"] = test_df["text"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
	## Number of title case words in the text ##
	train_df["num_words_title"] = train_df["text"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))
	test_df["num_words_title"] = test_df["text"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))
	## Average length of the words in the text ##
	train_df["mean_word_len"] = train_df["text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
	test_df["mean_word_len"] = test_df["text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))

###########A simple XGBoost model####################
def runXGB(train_X, train_y, test_X, test_y=None, test_X2=None, seed_val=0, child=1, colsample=0.5):
    param = {}
    param['objective'] = 'binary:logistic'
    param['eta'] = 0.01
    param['max_depth'] = 3
    param['silent'] = 1
    param['num_class'] = 3
    param['eval_metric'] = "auc"  ###multiclass logloss 
    param['min_child_weight'] = child
    param['subsample'] = 0.5
    param['colsample_bytree'] = colsample
    param['seed'] = seed_val
    num_rounds = 10000

    plst = list(param.items())
    xgtrain = xgb.DMatrix(train_X, label=train_y)

    if test_y is not None:
        xgtest = xgb.DMatrix(test_X, label=test_y)
        watchlist = [ (xgtrain,'train'), (xgtest, 'test') ]
        model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=50, verbose_eval=2000)
    else:
        xgtest = xgb.DMatrix(test_X)
        model = xgb.train(plst, xgtrain, num_rounds)
    
    #cvresult = xgb.cv(plst,xgtrain,num_rounds,nfold=5,metrics='merror',early_stopping_rounds=50,show_progress=False)
    pred_test_y = model.predict(xgtest, ntree_limit = model.best_ntree_limit)
    if test_X2 is not None:
        xgtest2 = xgb.DMatrix(test_X2)
        pred_test_y2 = model.predict(xgtest2, ntree_limit = model.best_ntree_limit)
    return pred_test_y, pred_test_y2, model

def runMNB(train_X, train_y, test_X, test_y, test_X2):
    model = naive_bayes.MultinomialNB()
    model.fit(train_X, train_y)
    pred_test_y = model.predict_proba(test_X)
    pred_test_y2 = model.predict_proba(test_X2)
    return pred_test_y, pred_test_y2, model

####### k-fold cross validation###############
def cv_xgb(train_X,train_y,train_df,test_X):
    kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2017)
    cv_scores = []
    pred_full_test = 0
    pred_train = np.zeros([train_df.shape[0], 2])
    for dev_index, val_index in kf.split(train_X):
        dev_X, val_X = train_X.loc[dev_index], train_X.loc[val_index]
        dev_y, val_y = train_y[dev_index], train_y[val_index]
        pred_val_y, pred_test_y, model = runXGB(dev_X, dev_y, val_X, val_y, test_X)
        pred_full_test = pred_full_test + pred_test_y
        pred_train[val_index,:] = pred_val_y
		#cv_scores.append(metrics.log_loss(val_y, pred_val_y))
        cv_scores.append(metrics.roc_auc_score(val_y, pred_val_y[:,1]))
    pred_full_test = pred_full_test / 5.
    return pred_full_test,cv_scores

def cv_mnb(train_df,train_tfidf,test_tfidf):
    kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2017)
    cv_scores = []
    pred_full_test = 0
	#pred_train = np.zeros([train_df.shape[0], 2])
    pred_train = np.zeros([train_df.shape[0],2])
    for dev_index, val_index in kf.split(train_df):
        dev_X, val_X = train_tfidf[dev_index], train_tfidf[val_index]
        dev_y, val_y = train_y[dev_index], train_y[val_index]
        pred_val_y, pred_test_y, model = runMNB(dev_X, dev_y, val_X, val_y, test_tfidf)
        pred_full_test = pred_full_test + pred_test_y
        pred_train[val_index,:] = pred_val_y
		#cv_scores.append(metrics.log_loss(val_y, pred_val_y))
        cv_scores.append(metrics.roc_auc_score(val_y, pred_val_y[:,1]))
    pred_full_test = pred_full_test / 5.
    return pred_full_test,cv_scores,pred_train

######Text Based Features#######
### Fit transform the tfidf vectorizer ###
def tfidf_word(train_df,test_df):
	#tfidf_vec = TfidfVectorizer(stop_words='english', ngram_range=(1,1),use_idf=True)    
    tfidf_vec = TfidfVectorizer(stop_words='english', ngram_range=(1,1),use_idf=True)    
    full_tfidf = tfidf_vec.fit_transform(train_df['text'].values.tolist() + test_df['text'].values.tolist())
    train_tfidf = tfidf_vec.transform(train_df['text'].values.tolist())
    test_tfidf = tfidf_vec.transform(test_df['text'].values.tolist())
    #Naive Bayes on Word Tfidf Vectorizer:
    pred_full_test, cv_scores,pred_train = cv_mnb(train_df,train_tfidf,test_tfidf)
    print("Naive Bayes on Word Tfidf Vectorizer")
    print("Mean cv score : ", np.mean(cv_scores))
    #add the predictions as new features
    train_df["nb_tfidf_word_0"] = pred_train[:,0]
    train_df["nb_tfidf_word_1"] = pred_train[:,1]
    test_df["nb_tfidf_word_0"] = pred_full_test[:,0]
    test_df["nb_tfidf_word_1"] = pred_full_test[:,1]
	###SVD on word TFIDF:
    n_comp = 20 ##recommended value for LSA
    svd_obj = TruncatedSVD(n_components=n_comp, algorithm='arpack')
    svd_obj.fit(full_tfidf)
    train_svd = pd.DataFrame(svd_obj.transform(train_tfidf))
    test_svd = pd.DataFrame(svd_obj.transform(test_tfidf))
    
    train_svd.columns = ['svd_word_'+str(i) for i in range(n_comp)]
    test_svd.columns = ['svd_word_'+str(i) for i in range(n_comp)]
    train_df = pd.concat([train_df, train_svd], axis=1)
    test_df = pd.concat([test_df, test_svd], axis=1)
    return train_df,test_df
#Naive Bayes on Word Count Vectorizer:
### Fit transform the count vectorizer ###
def CountV_word(train_df,test_df):
	#tfidf_vec = CountVectorizer(stop_words='english', ngram_range=(1,1))
    tfidf_vec = CountVectorizer(stop_words='english', ngram_range=(1,1))
    tfidf_vec.fit(train_df['text'].values.tolist() + test_df['text'].values.tolist())
    train_tfidf = tfidf_vec.transform(train_df['text'].values.tolist())
    test_tfidf = tfidf_vec.transform(test_df['text'].values.tolist())
    #build Multinomial NB model using count vectorizer based features.
    pred_full_test,cv_scores,pred_train = cv_mnb(train_df,train_tfidf,test_tfidf)    
    print("Multinomial NB model using count vectorizer based features")
    print("Mean cv score : ", np.mean(cv_scores))

    # add the predictions as new features #
    train_df["nb_cvec_0"] = pred_train[:,0]
    train_df["nb_cvec_1"] = pred_train[:,1]
    test_df["nb_cvec_0"] = pred_full_test[:,0]
    test_df["nb_cvec_1"] = pred_full_test[:,1]

#Naive Bayes on Character Count Vectorizer
### Fit transform the tfidf vectorizer ###
def CountV_char(train_df,test_df):
    tfidf_vec = CountVectorizer(ngram_range=(1,1), analyzer='char')
    tfidf_vec.fit(train_df['text'].values.tolist() + test_df['text'].values.tolist())
    train_tfidf = tfidf_vec.transform(train_df['text'].values.tolist())
    test_tfidf = tfidf_vec.transform(test_df['text'].values.tolist())
    pred_full_test,cv_scores,pred_train = cv_mnb(train_df,train_tfidf,test_tfidf)
    print("Fit transform the tfidf vectorizer")
    print("Mean cv score : ", np.mean(cv_scores))

    # add the predictions as new features #
    train_df["nb_cvec_char_0"] = pred_train[:,0]
    train_df["nb_cvec_char_1"] = pred_train[:,1]
    test_df["nb_cvec_char_0"] = pred_full_test[:,0]
    test_df["nb_cvec_char_1"] = pred_full_test[:,1]

#Naive Bayes on Character Tfidf Vectorizer:
### Fit transform the tfidf vectorizer ###
def tfidf_char(train_df,test_df):
    tfidf_vec = TfidfVectorizer(ngram_range=(1,1), analyzer='char')
    full_tfidf = tfidf_vec.fit_transform(train_df['text'].values.tolist() + test_df['text'].values.tolist())
    train_tfidf = tfidf_vec.transform(train_df['text'].values.tolist())
    test_tfidf = tfidf_vec.transform(test_df['text'].values.tolist())
    pred_full_test,cv_scores,pred_train = cv_mnb(train_df,train_tfidf,test_tfidf)
    print("Fit transform the tfidf vectorizer")
    print("Mean cv score : ", np.mean(cv_scores))
    # add the predictions as new features #
    train_df["nb_tfidf_char_0"] = pred_train[:,0]
    train_df["nb_tfidf_char_1"] = pred_train[:,1]
    test_df["nb_tfidf_char_0"] = pred_full_test[:,0]
    test_df["nb_tfidf_char_1"] = pred_full_test[:,1]
    ##SVD on Character TFIDF:
    n_comp = 20
=======

train_col=['company','label','text']
train_file = "/data/scratch/lingrui/sec_temp/workspace/test_200.csv"
test_col=['company','label','text']
test_file = "/data/scratch/lingrui/sec_temp/workspace/test_200.csv"
def main():
    #Set seed for reproducibility
    np.random.seed(0)
    os.system('date')
    print("Loading data...")
    #Load the data from the CSV files
    training_data = load_data(train_file,train_col)
    test_data = load_data(test_file,test_col)
    train_y = training_data['label'].values.astype(int)
    test_id = test_data['company'].values
    N = train_y.shape[0]
    N1 = np.sum(train_y == 1)
    N0 = N - N1
    model_xgb = xgb.XGBClassifier(scale_pos_weight=1.0*N0/N1,**param)
    model_mnb = naive_bayes.MultinomialNB()
    #Meta feature analysis
    print ('Meta feature statistical...')
    os.system('date')
    metaFeature(training_data)
    metaFeature(test_data)
    print ("TFIDF Vectorize data ...")
    os.system('date')
    full_tfidf_c,train_tfidf_c,test_tfidf_c = TfidfV(training_data,test_data,'char')
    #print(type(full_tfidf_c))
    full_tfidf_w,train_tfidf_w,test_tfidf_w = TfidfV(training_data,test_data,'word')
    print ("TFIDF MNB model on character")
    os.system('date')
    pred_train,pred_test=cv(model_mnb,train_tfidf_c,train_y,test_tfidf_c)  
    training_data['nb_tfidf_c_0'] = pred_train[:,0]
    training_data['nb_tfidf_c_1'] = pred_train[:,1]
    test_data['nb_tfidf_c_0'] = pred_test[:,0]
    test_data['nb_tfidf_c_1'] = pred_test[:,1]
    '''
    print ("SVD on TFIDF character")
    os.system('date')
    training_data,test_data = SVD(training_data,test_data,'char',full_tfidf_c,train_tfidf_c,test_tfidf_c)
    '''
    print ("TFIDF MNB model on word")
    os.system('date')
    pred_train,pred_test=cv(model_mnb,train_tfidf_w,train_y,test_tfidf_w)  
    training_data['nb_tfidf_w_0'] = pred_train[:,0]
    training_data['nb_tfidf_w_1'] = pred_train[:,1]
    test_data['nb_tfidf_w_0'] = pred_test[:,0]
    test_data['nb_tfidf_w_1'] = pred_test[:,1]
    print ("SVD on TFIDF word")
    os.system('date')
    training_data,test_data = SVD(training_data,test_data,'word',full_tfidf_w,train_tfidf_w,test_tfidf_w)

    print ('Count Vectorize data...')
    os.system('date')
    train_count_c,test_count_c = CountV(training_data,test_data,'char') 
    train_count_w,test_count_w = CountV(training_data,test_data,'word') 

    print ("CountV MNB model on character")
    os.system('date')
    pred_train,pred_test=cv(model_mnb,train_count_c,train_y,test_count_c)  
    training_data['nb_countv_c_0'] = pred_train[:,0]
    training_data['nb_countv_c_1'] = pred_train[:,1]
    test_data['nb_countv_c_0'] = pred_test[:,0]
    test_data['nb_countv_c_1'] = pred_test[:,1]

    print ("CountV MNB model on word")
    os.system('date')
    pred_train,pred_test=cv(model_mnb,train_count_w,train_y,test_count_w)  
    training_data['nb_countv_w_0'] = pred_train[:,0]
    training_data['nb_countv_w_1'] = pred_train[:,1]
    test_data['nb_countv_w_0'] = pred_test[:,0]
    test_data['nb_countv_w_1'] = pred_test[:,1]
    os.system('date')
    print ('Xgboost...')
    cols_to_drop = ['company','text','label']
    train_X = training_data.drop(cols_to_drop, axis=1).as_matrix()
    test_X = test_data.drop(cols_to_drop, axis=1).as_matrix()
    pred_X,pred_x=cv(model_xgb,train_X,train_y,test_X)  

    os.system('date')
    print ('Writing prediction to prediction.csv')
    #train_X_df = pd.DataFrame(train_X)
    train_X_df = pd.DataFrame(training_data.drop(cols_to_drop,axis=1))
    train_X_df.to_csv("xgb_input.csv",index=False)
    out_df = pd.DataFrame(pred_x[:,1])
    out_df.columns = ['label']
    out_df.insert(0, 'company', test_id)
    out_df.to_csv("./prediction.csv", index=False)
    print ('Finished')
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
    n_comp = 100  #value for LSA
>>>>>>> e57edcf14b9a038c8a7fb13cff6db5d5b1e831d4
    svd_obj = TruncatedSVD(n_components=n_comp, algorithm='arpack')
    svd_obj.fit(full_tfidf)
    train_svd = pd.DataFrame(svd_obj.transform(train_tfidf))
    test_svd = pd.DataFrame(svd_obj.transform(test_tfidf))
        
    train_svd.columns = ['svd_'+split_type+'_'+str(i) for i in range(n_comp)]
    test_svd.columns = ['svd_'+split_type+'_'+str(i) for i in range(n_comp)]
    train_df = pd.concat([train_df, train_svd], axis=1)
    test_df = pd.concat([test_df, test_svd], axis=1)
    return train_df,test_df

def cv(model,X,Y,x):
    K = 5 
    kf = KFold(n_splits=K, shuffle=True, random_state=2017)
    i = 0 
    for train, val in kf.split(X):
        i += 1
        model.fit(X[train, :], Y[train])
        pred = model.predict_proba(X[val,:])[:,1]
        print("auc %d/%d:" % (i,K),metrics.roc_auc_score(Y[val],pred))
    model.fit(X,Y)
    #if model == 'model_xgb':
    
    if re.match('XGB',str(model)):
        f = open ("./feature.txt","w+")
        #print (model.feature_importances_,file = f)
        print(pd.DataFrame(model.feature_importances_, columns=['importance']),file = f)
    print('Predicting...')
    Y_pre = model.predict_proba(X)
    y_pre = model.predict_proba(x)
    return Y_pre,y_pre

if __name__ == '__main__':
    main()
