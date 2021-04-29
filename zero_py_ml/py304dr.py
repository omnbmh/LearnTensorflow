# -8- coding:utf-8 -8-
"""
使用 matplotlib 绘图指令 绘图
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


def dr_xtyp(_dat):
    for xss in plt.style.available:
        plt.style.use(xss)
        print(xss)
        plt.plot(_dat['open'])
        plt.plot(_dat['close'])
        plt.plot(_dat['high'])
        plt.plot(_dat['low'])
        fss = "plt_image_files/stk001_" + xss + ".png"
        plt.savefig(fss)
        plt.show()


df = pd.read_csv("dat/kline_btcusdt_15min.csv", encoding='utf-8')
d30 = df[0:30]
dr_xtyp(d30)
