from __future__ import print_function
import re 
xgb = 'XGBClassifier'
if re.match('XGB',xgb):
    f = open ("./feature.txt","w+")
    print ("haha",file = f)

