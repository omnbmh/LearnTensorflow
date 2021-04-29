# -8- coding:utf-8 -8-

import pandas as pd

df = pd.read_csv("dat/ccpp.csv", index_col=False)
print(df.tail())
print(df.describe())
