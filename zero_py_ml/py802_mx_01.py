# -8- coding:utf-8 -8-

from zero_py_ml import xai

x_train, x_test, y_train, y_test = xai.ai_data_rd('dat/ccpp_', k0=10, fgPr=True)

# 2
funcSign = 'line'
# tim0 = arrow.now()
dacc, df9 = xai.mx_fun010(funcSign, x_train, x_test, y_train, y_test, 5, False, True)

funcSign = 'log'
dacc, df9 = xai.mx_fun010(funcSign, x_train, x_test, y_train, y_test, 5, False, True)
