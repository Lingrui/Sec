#!/usr/bin/env python
#coding=utf-8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import pickle
import numpy as np
from scipy.misc import imread
import matplotlib.pyplot as plt

from wordcloud import WordCloud, ImageColorGenerator

DICT = {}  
i = 0 
with open('TFIDF_dictionary_word.txt','r') as dic:
    for l in dic:
        word = l.strip()
        DICT[i] = word
        i += 1 
c = np.load("TFIDF_SVD.npy")
print(c.shape)
# read the mask image 
#bg = imread('/shared/s2/users/jwang/photo/P1.jpg')
bg = imread('princess.png')
wc = WordCloud(background_color='white',
        mask= bg,
        max_font_size=100,
        random_state=42)

image_colors = ImageColorGenerator(bg)

k = 0
for line in c:
    name = {}    
    for s in line.argsort()[-30:][::-1]:
        name[DICT[s]] = line[s]
    '''
    print (k)
    for key,values in name.items():
        print (key,values)	
    '''
    wc.generate_from_frequencies(name)
	#plt.imshow(wc)
    #plt.axis("off")
    #plt.figure()
    plt.imshow(wc.recolor(color_func=image_colors))
    #plt.axis("off")
    #plt.show
    filename = 'SVD_'+str(k)+'.png'
	#wc.to_file("haha.png")
    wc.to_file(filename)
    k += 1
'''

plt.imshow(wc)
plt.axis("off")

plt.figure()
'''
