#!/usr/bin/env python 
from __future__ import print_function 
from __future__ import division
from __future__ import absolute_import 

import requests 
import bs4
import sys,os

_CIK_URI = 'http://www.sec.gov/cgi-bin/browse-edgar' \
    '?action=getcompany&CIK={s}&count=10&output=xml'

inputfile = str((sys.argv[1]))    
outputfile = str((sys.argv[2]))    

def get_cik(symbol):
    """
    Retrieves the CIK identifier of a given security from the SEC based on that
    security's market symbol (i.e. "stock ticker").
    :param symbol: Unique trading symbol (e.g. 'NVDA')
    :return: A corresponding CIK identifier (e.g. '1045810')
    """

    response = requests.get(_CIK_URI.format(s=symbol))
    page_data = bs4.BeautifulSoup(response.text, "html.parser")
    try:
        cik = page_data.companyinfo.cik.string
        return cik
    except:
        return 'NA'

with open (inputfile, "r") as f :
    for i in f.readlines():
        i = str(i.strip())
        with open(outputfile,'a') as wf:
            print(i,"\t",get_cik(i),file=wf) 

