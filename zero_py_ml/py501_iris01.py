# -8- coding: utf-8 -8-
import pandas as pd

fss = 'dat/iris.data'
df = pd.read_csv(fss, sep=',', index_col=False, header=None)
df.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]

print('\n#1 df')
print(df.tail())
print(df.describe())

df10 = df['class'].value_counts()
print('\n#2 xname')
print(df10)
