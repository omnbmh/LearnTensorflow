# -8- coding:utf-8 -8-
import pandas as pd

pd.set_option('display.width', 450)

fss = 'data/600663.csv'
df = pd.read_csv(fss, index_col=0)  # index_col 索引列
print(df.tail())
