#!/usr/bin/env python 
from __future__ import print_function
import csv
import pickle
f = open('/data/scratch/lingrui/sec_temp/workspace/test_all.csv',"r")

reader = csv.reader(f)
result = {}
for item in reader:
    result[reader.line_num - 1] = item
pickle.dump(reader,open('/data/scratch/lingrui/sec_temp/workspace/test_all.dat',"w"))

#h = pickle.load(open('pickled.dat','r'))
#print(h)
