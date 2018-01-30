#!/usr/bin/env python
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
with open('test.dictionary.txt','r') as dic:
    for l in dic:
        word = l.strip()
        DICT[i] = word
        i += 1 
#print (DICT[1139]) 
c = np.load("test_SVD.npy")
# read the mask image 
bg = imread('/shared/s2/users/jwang/photo/P1.jpg')
wc = WordCloud(background_color='white',
        mask= bg,
        max_font_size=40,
        random_state=42)

image_colors = ImageColorGenerator(bg)

k = 0
prefix = 'frequency'
for line in c:
    for s in line.argsort()[-20:][::-1]:
        name = prefix + str(k)
        name = {}    
        name[DICT[s]] = line[s]
        wc.generate_from_frequencies(name)
        plt.imshow(wc)
        plt.axis("off")
        wc.to_file("haha.png")
    k += 1
'''

plt.imshow(wc)
plt.axis("off")

plt.figure()
'''
