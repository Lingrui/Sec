#!/usr/bin/env python 
from __future__ import print_function
import xlrd
import csv
import os.path

with open('../preprocess/EPC-16.csv','wb') as myCsvfile:
    #define a writer 
    wr = csv.writer(myCsvfile,delimiter="\t")

    #open the xlsx file 
    myfile = xlrd.open_workbook('../raw_data/EPC-16.xlsx','wb')
    #get a sheet
    for mysheet in myfile.sheets():
    
        #write the rows
        for rownum in xrange(mysheet.nrows):
            #if mysheet.row_values(rownum)[-1].endswith(r'\r'):
            #mysheet.row_values(rownum)[-1].replace(r'\r','')
            wr.writerow([unicode(s).encode('utf-8').replace("\n","") for s in mysheet.row_values(rownum)]) 
