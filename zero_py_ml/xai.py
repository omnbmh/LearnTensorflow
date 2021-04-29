# -8- coding:utf-8 -8-

"""
封装sklearn机器学习算法
"""
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
# scikit-learn 在 0.23 版本之后已经将 joblib 移除，需要直接安装 joblib
# from sklearn.externals import joblib
import joblib


# 分割数据
def ai_data_split(df, xlst, ysgn, ftg0, fgPr=False):
    x, y = df[xlst], df[ysgn]
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)
    # 'dat/ccpp_'
    x_train.to_csv(ftg0 + "x_train.csv", index=False)
    x_test.to_csv(ftg0 + "x_test.csv", index=False)
    y_train.to_csv(ftg0 + "y_train.csv", index=False, header=True)
    y_test.to_csv(ftg0 + "y_test.csv", index=False, header=True)
    if fgPr:
        print('\nx_train')
        print(x_train.tail())
        print('\nx_test')
        print(x_test.tail())
        print('\ny_train')
        print(y_train.tail())
        print('\ny_test')
        print(y_test.tail())


# 数据读取
def ai_data_rd(fsr0, k0=1, fgPr=False):
    x_train = pd.read_csv(fsr0 + "x_train.csv", index_col=False)
    x_test = pd.read_csv(fsr0 + "x_test.csv", index_col=False)
    y_train = pd.read_csv(fsr0 + "y_train.csv", index_col=False)
    y_test = pd.read_csv(fsr0 + "y_test.csv", index_col=False)
    # ysgn
    ysgn = y_train.columns[0]
    y_train[ysgn] = round(y_train[ysgn] * k0).astype(int)
    y_test[ysgn] = round(y_test[ysgn] * k0).astype(int)

    if fgPr:
        print('\nx_train')
        print(x_train.tail())
        print('\nx_test')
        print(x_test.tail())
        print('\ny_train')
        print(y_train.tail())
        print('\ny_test')
        print(y_test.tail())
    return x_train, x_test, y_train, y_test


# 效果评测函数
def ai_acc_xed(df9, ky0=5, bgDebug=False):
    # 1
    ny_test, ny_pred = len(df9['y_test']), len(df9['y_pred'])
    df9['ysub'] = df9['y_test'] - df9['y_pred']
    df9['ysub2'] = np.abs(df9['ysub'])

    # 2
    df9['y_test_div'] = df9['y_test']
    df9.loc[df9['y_test'] == 0, 'y_test_div'] = 0.00001
    df9['ysubk'] = (df9['ysub2'] / df9['y_test_div']) * 100
    dfk = df9[df9['ysubk'] < ky0]
    dsum = len(dfk['y_pred'])
    dacc = dsum / ny_test * 100

    # 3
    if bgDebug:
        print('\nai_acc_xed')
        print(df9.head())
        y_test, y_pred = df9['y_test'], df9['y_pred']
        print('\ntest,{0}, npred,{1}, dsum,{2} '.format(ny_test, ny_pred, dsum))
        dmae = metrics.mean_absolute_error(y_test, y_pred)
        dmse = metrics.mean_squared_error(y_test, y_pred)
        drmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
        print("acc-kok: {0:.2f}%, MAE:{1:.2f}, MSE:{2:.2f}, RMSE:{3:.2f}".format(dacc, dmae, dmse, drmse))
    return dacc


# 线形回归算法
def mx_line(train_x, train_y):
    # LinearRegression(fit_intercept=True,normalize=False,copy_X=True,n_jobs=1)
    mx = LinearRegression()
    mx.fit(train_x, train_y)
    return mx


# 逻辑回归算法
def mx_log(train_x, train_y):
    mx = LogisticRegression(penalty='l2')
    mx.fit(train_x, train_y)
    return mx


# 多项式朴素贝叶斯算法
def mx_bayes(train_x, train_y):
    mx = MultinomialNB(alpha=0.01)
    mx.fit(train_x, train_y)
    return mx


# KNN近邻算法
def mx_knn(train_x, train_y):
    mx = KNeighborsClassifier()
    mx.fit(train_x, train_y)
    return mx


# 随机森林算法
def mx_forest(train_x, train_y):
    mx = RandomForestClassifier(n_estimators=8)
    mx.fit(train_x, train_y)
    return mx


# 决策树算法
def mx_dtree(train_x, train_y):
    mx = DecisionTreeClassifier()
    mx.fit(train_x, train_y)
    return mx


# GBDT迭代决策树算法
def mx_gbdt(train_x, train_y):
    mx = GradientBoostingClassifier(n_estimators=200)
    mx.fit(train_x, train_y)
    return mx


def mx_svm(train_x, train_y):
    mx = SVC(kernel='rbf', probability=True)
    mx.fit(train_x, train_y)
    return mx


mxfuncSgn = {'line': mx_line,
             'log': mx_log,
             'svm': mx_svm}


def mx_fun010(funcSign, x_train, x_test, y_train, y_test, yk0=5, fgInt=False, fgDebug=False):
    """

    :param funcSign:
    :param x_train:
    :param x_test:
    :param y_train:
    :param y_test:
    :param yk0: 结果误差 k 值 默认是5 标识5% 整数设置为1
    :param fgInt:
    :param fgDebug:
    :return:
    """
    # 1 搞到模型
    df9 = x_test.copy()
    mx_func = mxfuncSgn[funcSign]
    mx = mx_func(x_train.values, y_train.values)

    # 2 预测
    y_pred = mx.predict(x_test.values)
    df9['y_test'], df9['y_pred'] = y_test, y_pred

    # 3 整数化 CCPP 不需要
    if fgInt:
        df9['y_predsr'] = df9['y_pred']
        df9['y_pred'] = round(df9['y_predsr']).astype(int)

    dacc = ai_acc_xed(df9, yk0, fgDebug)
    if fgDebug:
        print("@func name: ", mx_func.__name__)

    # 6
    print("@mx:mxsum, kok:{0:.2f}%".format(dacc))
    return dacc, df9

def mx_fun8mx(mx, x_test, y_test, yk0=5, fgInt=False, fgDebug=False):
    """

    :param funcSign:
    :param x_train:
    :param x_test:
    :param y_train:
    :param y_test:
    :param yk0: 结果误差 k 值 默认是5 标识5% 整数设置为1
    :param fgInt:
    :param fgDebug:
    :return:
    """
    # 1 搞到模型
    df9 = x_test.copy()

    # 2 预测
    y_pred = mx.predict(x_test.values)
    df9['y_test'], df9['y_pred'] = y_test, y_pred

    # 3 整数化 CCPP 不需要
    if fgInt:
        df9['y_predsr'] = df9['y_pred']
        df9['y_pred'] = round(df9['y_predsr']).astype(int)

    # 4
    dacc = ai_acc_xed(df9, yk0, fgDebug)

    # 6
    print("@mx:mxsum, kok:{0:.2f}%".format(dacc))
    return dacc, df9

# 保存模型
def ai_f_mxWr(ftg, funcSgn, x_train, y_train):
    mx_func = mxfuncSgn[funcSgn]
    mx = mx_func(x_train.values, y_train.values)
    joblib.dump(mx, ftg)


if __name__ == "__main__":
    # ai_data_split(pd.read_csv('dat/ccpp.csv', index_col=False)
    #               , ['AT', 'V', 'AP', 'RH'], 'PE'
    #               , 'dat/ccpp_', True)
    x_train, x_test, y_train, y_test = ai_data_rd('dat/ccpp_', k0=10, fgPr=True)
