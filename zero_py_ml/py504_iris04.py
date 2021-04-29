# -8- coding:utf-8 -8-

import pandas as pd
from sklearn.linear_model import LinearRegression

x_train = pd.read_csv('dat/x_train.csv', index_col=False)
y_train = pd.read_csv('dat/y_train.csv', index_col=False)

print(x_train.tail())
print(y_train.tail())

print("建模")

# mx = zai.mx_line(x_train.values, y_train.values)

def mx_line(train_x, train_y):
    # LinearRegression(fit_intercept=True,normalize=False,copy_X=True,n_jobs=1)
    mx = LinearRegression()
    mx.fit(train_x, train_y)
    return mx


mx = mx_line(x_train.values, y_train.values)

x_test = pd.read_csv('dat/x_test.csv', index_col=False)
df9 = x_test.copy()
print(x_test.tail())

print('预测')
y_pred = mx.predict(x_test.values)
df9['y_predsr'] = y_pred

y_test = pd.read_csv('dat/y_test.csv', index_col=False)
print(y_test.tail())

df9['y_test'] = y_test
df9['y_pred'] = round(df9['y_predsr']).astype(int)
print(df9.tail())
