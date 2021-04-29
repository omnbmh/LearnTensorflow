# -8- coing:utf-8
"""
存储算法模型
"""
from zero_py_ml import xai
import joblib

# 1

xlst, ysgn = ["AT", "V", "AP", "RH"], "PE"

x_train, x_test, y_train, y_test = xai.ai_data_rd("dat/ccpp_")

funcSgn = "svm"
ftg = "dat/ccpp_svm.pkl"

# 2 模型存起来
# xai.ai_f_mxWr(ftg, funcSgn, x_train, y_train)

# 3 读取模型

mx = joblib.load(ftg)

xai.mx_fun8mx(mx, x_test, y_test)
