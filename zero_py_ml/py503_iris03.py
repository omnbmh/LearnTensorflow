# -8- coding:utf-8 -8-
import pandas as pd
from sklearn.model_selection import train_test_split

fss = 'dat/iris_20200817.csv'
df = pd.read_csv(fss, sep=',', index_col=False, header=None)
df.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class", "xid"]

print(df.tail())

# 拆分 x,y

xlst, ysgn = ["sepal_length", "sepal_width", "petal_length", "petal_width"], 'xid'

x, y = df[xlst], df[ysgn]
print('x')
print(x.tail())

print('y')
print(y.tail())

print('#4')
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)
x_test.index.name = 'xid'
y_test.index.name = 'xid'

print('tpye x_train ', type(x_train))
print('tpye x_test ', type(x_test))
print('tpye y_train ', type(y_train))
print('tpye y_test ', type(y_test))

print("# save csv")
x_train.to_csv('dat/x_train.csv', index=False)
x_test.to_csv('dat/x_test.csv', index=False)
y_train.to_csv('dat/y_train.csv', index=False, header=True)
y_test.to_csv('dat/y_test.csv', index=False, header=True)
