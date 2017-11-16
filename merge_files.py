#!/usr/bin/python 
from __future__ import print_function
from __future__ import division

import re
import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import string
prepro_dir = '/home/lcai/s2/sec_10k/preprocess/csv'
## Read the train and test dataset and check the top few lines ##
data_df = pd.read_table("/home/lcai/s2/sec_10k/preprocess/status",header=None)
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
            with open(fpath,'r') as f:
            #t = f.read()
                for line in f.readlines():
                    line = line.strip('\n')
            #t_all = t + t_all
            t_all = line + t_all
    texts.append(t_all)
data_df["text"] = texts

#########Print out training data######################
######################################################3
out_df = pd.DataFrame(data_df)
out_df.columns = ['company', 'lable', 'text']
out_df.to_csv("merged.csv", index=False)
#out_df.to_csv("xgb_countV_word.csv", index=False)

