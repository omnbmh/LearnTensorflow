# -8- coding:utf-8 -8-
import pandas as pd
from zero_py_ml import xai

"""
多项式朴素贝叶斯算法
"""

# 1. 读取数据

x_train = pd.read_csv('dat/x_train.csv', index_col=False)
y_train = pd.read_csv('dat/y_train.csv', index_col=False)

x_test = pd.read_csv('dat/x_test.csv', index_col=False)
y_test = pd.read_csv('dat/y_test.csv', index_col=False)

df9 = x_test.copy()


# 2. 建模
mx = xai.mx_bayes(x_train.values, y_train.values)

# 3. 预测
y_pred = mx.predict(x_test.values)
df9['y_predsr'] = y_pred
df9['test'] = y_test
df9['y_pred'] = round(df9['y_predsr']).astype(int)
print(df9.tail())

print(1 in df9['y_predsr'].values)
print(5 in df9['y_predsr'].values)
