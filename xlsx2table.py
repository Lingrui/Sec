#!/usr/bin/env python 
from __future__ import print_function
import xlrd
import csv
import sys,getopt
import re 

inputfile = str(sys.argv[1])
outputfile = str(sys.argv[2])

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
        sheet = 0 #initial sheet number    
        headrow = ['sheet_name','row_name','date','value']
        wr.writerow(headrow)
        #get a sheet
        for mysheet in myfile.sheets():
            tmp = []
            i = 0 
            sn = ''
            times = []
            #write the rows
            for rownum in xrange(mysheet.nrows):
                #mysheet.row_values(rownum)[-1].replace(r'\r','')
                #wr.writerow([unicode(s).encode('utf-8').replace("\n","") for s in mysheet.row_values(rownum)]) 
                tmp = [unicode(s).encode('utf-8').replace("\n","") for s in mysheet.row_values(rownum)]
                ## first sheet 
                if sheet == 0:
                    if i == 0:
                        sn = tmp[0]
                        for t in tmp[1:]:
                            #make sure the date
                            if re.match(r'^\w{3}.?\s?\d+,\s?\d+',t): 
                                times.append(t)
                    elif i == 1:
                        for t in tmp[1:]:
                            if re.match(r'^\w{3}.?\s?\d+,\s?\d+',t):
                                times.append(t)
                    else:
                        #if re.match(r'\d+',tmp[-1]): 
                        if tmp[1] != '' or re.match(r'\d+',tmp[-1]):
                            tmp.insert(0,sn) 
                            for 
                            for time in times:
                                wr.writerow(tmp)
                i += 1
            sheet += 1
