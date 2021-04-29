# -8- coding:utf-8 -8-

"""
本程序 评估器

数据集
https://www.tfimgs.cn/download.tensorflow.org/data/iris_test.csv
https://www.tfimgs.cn/download.tensorflow.org/data/iris_training.csv

tensorflow contrib 包
pip install tensorflow==1.15.3
"""

from tensorflow.contrib.learn.python.learn.datasets import base

# Data files
IRIS_TRAINING = "iris_training.csv"
IRIS_TEST = "iris_training.csv"

# Load datasets
training_set =