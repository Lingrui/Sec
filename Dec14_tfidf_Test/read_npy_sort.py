#!/usr/bin/env python
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import pickle
import numpy as np
c = np.load("test_SVD.npy")
for line in c:
    '''
    line.sort()
    print(line)
    '''
    print(line.argsort()[-10:][::-1])
#f = open ("Top10_words.txt","w+")
#    print(line.argsort()[-10:][::-1],file =f )
