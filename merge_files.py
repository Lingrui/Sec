#!/usr/bin/python 
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import  
# encoding=utf8 
import sys
reload(sys) 
sys.setdefaultencoding('utf8')

import re
import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import string
prepro_dir = '/home/lcai/s2/sec_10k/preprocess/csv'
#prepro_dir = '/home/lcai/s2/sec_10k/test/csv'
## Read the train and test dataset and check the top few lines ##
data_df = pd.read_table("/home/lcai/s2/sec_10k/preprocess/status",header=None)
#data_df = pd.read_table("/home/lcai/s2/sec_10k/test/status.test",header=None)
data_id = data_df[0]
data_y = data_df[1]
print("Number of rows in dataset : ",data_df.shape[0])
##################################################

i = 0 
texts = []
for name in data_id:
    t_all = ''
    for fname in sorted(os.listdir(prepro_dir)):
        if(re.match(name+'-',fname)):
            fpath = os.path.join(prepro_dir,fname)
            f = open(fpath,'r')
            #t = f.read()
            for line in f.readlines():
                line = line.strip('\n')
                #line.decode().encode('utf-8').replace("\r","")
                line = unicode(line,encoding='utf-8')
                t_all = line.replace("\r","") + t_all
    texts.append(t_all)
data_df["text"] = texts

#########Print out training data######################
######################################################3
out_df = pd.DataFrame(data_df)
out_df.columns = ['company', 'label', 'text']
out_df.to_csv("merged.csv", index=False)
#out_df.to_csv("xgb_countV_word.csv", index=False)

