#!/usr/bin/env python 
from __future__ import print_function
import xlrd
import csv
import sys,getopt

inputfile = str(sys.argv[1])
outputfile = str(sys.argv[2])
'''
try:
    opts,args = getopt.getopt(sys.argv,'hi:o:',['ifile=','ofile='])
except getopt.GetoptError:
    print('xlsx2csv.py -i <inputfile> -o  <outputfile>')
    sys.exit(2)
for opt,arg in opts:
    if opt == '-h':
        print('xlsx2csv.py -i <inputfile> -o <outputfile>')
        sys.exit()
    elif opt in ('-i','--ifile'):
        inputfile = arg
    elif opt in ('-o','--ofile'):
        outputfile = arg
'''
#with open('../preprocess/EPC-16.csv','wb') as myCsvfile:
with open(outputfile,'wb') as myCsvfile:
    #define a writer 
    wr = csv.writer(myCsvfile,delimiter="\t")

    #open the xlsx file 
    #myfile = xlrd.open_workbook('../raw_data/EPC-16.xlsx','wb')
    try:
        myfile = xlrd.open_workbook(inputfile,'wb')
    #except IOError:
    except:
        print("Error",inputfile)
    else:
    #get a sheet
        for mysheet in myfile.sheets():
    
            #write the rows
            for rownum in xrange(mysheet.nrows):
                #mysheet.row_values(rownum)[-1].replace(r'\r','')
                wr.writerow([unicode(s).encode('utf-8').replace("\n","") for s in mysheet.row_values(rownum)]) 
