#!/usr/bin/python 
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import sys
import re
import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import nltk
from nltk.corpus import stopwords
import string
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn import ensemble, metrics, model_selection, naive_bayes

eng_stopwords = set(stopwords.words("english"))
pd.options.mode.chained_assignment = None
## Read the train and test dataset and check the top few lines ##
train_df = pd.DataFrame(columns=['company','label','text'])
for pred_df in pd.read_csv("/data/scratch/lingrui/sec_temp/workspace/test_all.csv",header=0,chunksize = 500):
    train_df = pd.concat([train_df,pred_df],axis=0)
train_df.to_csv('haha.csv', index=False)
test_df = train_df
#train_df = pd.read_csv("/data/scratch/lingrui/sec_temp/workspace/test_all.csv",header=0)
#test_df = pd.read_csv("/data/scratch/lingrui/sec_temp/workspace/test_all.csv",header=0)
train_y = train_df['label'].values.astype(int)

print(train_y.shape)
train_id = train_df['company'].values
test_id = test_df['company'].values

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
    param['eta'] = 0.001
    param['max_depth'] = 3
    param['silent'] = 1
    #param['num_class'] = 3
    param['eval_metric'] = "auc"  ###multiclass logloss 
    param['min_child_weight'] = child
    param['subsample'] = 0.5
    param['colsample_bytree'] = colsample
    param['seed'] = seed_val
    num_rounds = 1000

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
    pred_train = np.zeros([train_df.shape[0], 1])
    for dev_index, val_index in kf.split(train_X):
        dev_X, val_X = train_X.loc[dev_index], train_X.loc[val_index]
        dev_y, val_y = train_y[dev_index], train_y[val_index]
        pred_val_y, pred_test_y, model = runXGB(dev_X, dev_y, val_X, val_y, test_X)
        #print ("pred_val_y: ",pred_val_y)
        pred_full_test = pred_full_test + pred_test_y
        #pred_train[val_index,:] = pred_val_y
		#cv_scores.append(metrics.log_loss(val_y, pred_val_y))
        cv_scores.append(metrics.roc_auc_score(val_y, pred_val_y))
    pred_full_test = pred_full_test / 5.
    #return pred_full_test,cv_scores
    return pred_val_y,cv_scores

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
    svd_obj = TruncatedSVD(n_components=n_comp, algorithm='arpack')
    svd_obj.fit(full_tfidf)
    train_svd = pd.DataFrame(svd_obj.transform(train_tfidf))
    test_svd = pd.DataFrame(svd_obj.transform(test_tfidf))
    
    train_svd.columns = ['svd_char_'+str(i) for i in range(n_comp)]
    test_svd.columns = ['svd_char_'+str(i) for i in range(n_comp)]
    train_df = pd.concat([train_df, train_svd], axis=1)
    test_df = pd.concat([test_df, test_svd], axis=1)
    return train_df,test_df
###XGBoost model:
metaFeature(train_df,test_df)
train_df,test_df = tfidf_word(train_df,test_df)
train_df,test_df = tfidf_char(train_df,test_df)
CountV_word(train_df,test_df)
CountV_char(train_df,test_df)

cols_to_drop = ['company', 'text']
train_X = train_df.drop(cols_to_drop+['label'], axis=1)
test_X = test_df.drop(cols_to_drop+['label'], axis=1)
#########Print out training data######################
train_X_df = pd.DataFrame(train_X)
train_X_df.to_csv("../workspace/test_all_xgb_input.csv",index=False)
#train_X_df.to_csv("countV_word.csv",index=False)
######################################################3
pred_full_test, cv_scores = cv_xgb(train_X,train_y,train_df,test_X)
print("cv scores : ", cv_scores)
out_df = pd.DataFrame(pred_full_test)
out_df.columns = ['label']
out_df.insert(0, 'company', test_id)
out_df.to_csv("../workspace/test_all_prediction.csv", index=False)
#out_df.to_csv("xgb_countV_word.csv", index=False)

