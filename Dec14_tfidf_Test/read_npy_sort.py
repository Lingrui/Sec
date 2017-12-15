#!/usr/bin/env python
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import pickle
import numpy as np
DICT = {}  
i = 0 
with open('test.dictionary.txt','r') as dic:
    for l in dic:
        word = l.strip()
        DICT[i] = word
        i += 1 
#print (DICT[1139]) 
c = np.load("test_SVD.npy")
for line in c:
    #print(line.argsort()[-10:][::-1])
    for s in line.argsort()[-3:][::-1]:

        print(DICT[s],'\t',line[s])
    print ()
