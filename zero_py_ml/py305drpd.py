# -8- coding:utf-8 -8-

"""
使用 pandas 内置的plot命令绘图
"""

import pandas as pd
import matplotlib.pyplot as plt


def dr_xtyp(_dat):
    for xss in plt.style.available:
        plt.style.use(xss)
        print(xss)

        _dat['open'].plot()  # 使用 pandas 的plot方法
        _dat['close'].plot()

        fss = "pd_image_files/ttk001_" + xss + "_pd.png"
        plt.savefig(fss)
        plt.show()


df = pd.read_csv("dat/kline_btcusdt_15min.csv", encoding='utf-8')
d30 = df[0:30]
dr_xtyp(d30)
