# -8- coding:utf-8 -8-8

"""
绘制金融数据K线图
"""

import pandas as pd
import plotly as py
import plotly.figure_factory as pyff

import zero_tf.xtools_draw as xdr

pd.set_option('display.width', 450)
# 函数别名
pyplt = py.offline.plot

# 2
xcod = '600663'
fss = 'data/' + xcod + '.csv'
df = pd.read_csv(fss, index_col=0)
df = df.sort_index(ascending=True)  # 正序
print(df.tail())

# 3
print('\n#3 plot --> /tmp/tmp_.html')
hdr, fss = 'K线图-' + xcod, '/tmp/tmp_.html'
df2 = df.tail(100)

xdr.drDF_cdl(df2, ftg=fss, m_title=hdr)
