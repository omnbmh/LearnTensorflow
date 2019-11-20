# -8- coding:utf-8 -8-

'''
使用模块化 来实现 神经网络 八股

前向传播就是搭建网络，设计网络结构
反向传播就是训练网络，优化网络参数
'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# 生成数据集 gen_data.py

def gen_data():
    # 基于 seed 生成随机数
    seed = 2
    rdm = np.random.RandomState(seed)
    # 随机生成300组 坐标点(x1,x2) 作为训练集
    X = rdm.randn(300, 2)
    # 判断坐标的平方和 小于 2 返回 1 其余返回 0
    Y_ = [int(x0 * x0 + x1 * x1 < 2) for (x0, x1) in X]
    # 遍历 Y 1 = 'red' 0 = 'blue'
    Y_color = [['red' if y else 'blue'] for y in Y_]

    # 形状整理
    X = np.vstack(X).reshape(-1, 2)  # 300 行 2 列
    Y_ = np.vstack(Y_).reshape(-1, 1)  # 300 行 1 列
    # print(X)
    # print(Y_)
    return X, Y_, Y_color


# 前向传播 forward.py
def get_weight(shape, regularizer):
    w = tf.Variable(tf.random_normal(shape), dtype=tf.float32)
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w


def get_bias(shape):
    return tf.Variable(tf.constant(0.01, shape=shape))


# 神经网络结构
def forward(x, regularizer):
    w1 = get_weight([2, 11], regularizer)
    b1 = get_bias([11])
    y1 = tf.nn.relu(tf.matmul(x, w1) + b1)

    w2 = get_weight([11, 1], regularizer)
    b2 = get_bias([1])
    y = tf.matmul(y1, w2) + b2  # 输出层 不用激活
    return y


# 后向传播 backward.py
STEPS = 40000  # 训练多少次
BATCH_SIZE = 30  # 每次训练样本数
LEARNING_RATE_BASE = 0.001  # 基础学习速率
LEARNING_RATE_DECAY = 0.999  #
REGULARIZER = 0.01  # 正则化


def backward():
    x = tf.placeholder(tf.float32, shape=(None, 2))
    y_ = tf.placeholder(tf.float32, shape=(None, 1))

    X, Y_, Y_color = gen_data()

    y = forward(x, REGULARIZER)
    global_step = tf.Variable(0, trainable=False)

    # 定义指数衰减学习率
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        300 / BATCH_SIZE,
        LEARNING_RATE_DECAY,
        staircase=True)

    loss_mse = tf.reduce_mean(tf.square(y - y_))
    loss_total = loss_mse + tf.add_n(tf.get_collection('losses'))

    # 定义反向传播方法 包含正则化
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss_total)

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        for i in range(STEPS):
            start = (i * BATCH_SIZE) % 300
            end = start + BATCH_SIZE

            sess.run(train_step, feed_dict={x: X[start:end], y_: Y_[start:end]})
            if i % 2000 == 0:
                loss_v = sess.run(loss_total, feed_dict={x: X, y_: Y_})
                print("After %d steps, loss is %f" % (i, loss_v))

        # 训练完成 看下效果
        xx, yy = np.mgrid[-3:3:.01, -3:3:.01]
        grid = np.c_[xx.ravel(), yy.ravel()]
        probs = sess.run(y, feed_dict={x: grid})
        probs = probs.reshape(xx.shape)
    plt.scatter(X[:, 0], X[:, 1], c=np.squeeze(Y_color))
    plt.contour(xx, yy, probs, levels=[.5])  # probs 为 0.5 的曲线
    plt.show()


if __name__ == '__main__':
    backward()
