# -8- coding:utf-8 -8-
import pandas as pd

fss = 'dat/iris.data'
df = pd.read_csv(fss, sep=',', index_col=False, header=None)
df.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]


df.loc[df['class'] == 'Iris-virginica', 'xid'] = 1
df.loc[df['class'] == 'Iris-setosa', 'xid'] = 2
df.loc[df['class'] == 'Iris-versicolor', 'xid'] = 3

df['xid'] = df['xid'].astype(int)
df.to_csv('dat/iris_20200817.csv', index=False)

print("\n#df")
print(df.tail())
print(df.describe())
