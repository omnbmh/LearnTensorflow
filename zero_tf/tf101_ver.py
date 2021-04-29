# -8- coding:utf-8 -8-
import tensorflow as tf
import tensorlayer as tl
import keras as ks
import nltk

import pandas as pd
import tushare as ts
import matplotlib as mpl

import plotly
import arrow

# 为了兼容 这样导入 tflearn
# import tflearn
tflearn = tf.contrib.learn

print('\n#1 tensorflow.ver: ', tf.__version__)
print('\n#2 tensorlayer.ver: ', tl.__version__)
print('\n#3 keras.ver: ', ks.__version__)
print('\n#4 nltk.ver: ', nltk.__version__)
print('\n#5 pandas.ver: ', pd.__version__)
print('\n#6 tushare.ver: ', ts.__version__)
print('\n#7 matplatlib.ver: ', mpl.__version__)
print('\n#8 plotly.ver: ', plotly.__version__)
print('\n#9 arrow.ver: ', arrow.__version__)
