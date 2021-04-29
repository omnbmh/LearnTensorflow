# -8- coding:utf-8 -8-
"""
Tick数据格式
"""
import pandas as pd
import plotly as py
import zero_tf.xtools_draw as xdr

pyplt = py.offline.plot

pd.set_option('display.width', 450)

# 2
fss = 'data/tk002645_2016-09-01.csv'
df = pd.read_csv(fss, index_col=False)
df = df.sort_values('time')
print(df.tail())

# 3
hdr, fss = 'Tick数据价格曲线图', '/tmp/tmp_.html'
df2 = df.tail(200)
xdr.drDF_tickX(df2, ftg=fss, m_title=hdr, sgnTim='time', sgnPrice='price')
