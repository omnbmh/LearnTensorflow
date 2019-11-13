# -8- coding:utf-8 -8-

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

BATCH_SIZE = 30
seed = 2
rdm = np.random.RandomState(seed)
# 300 个坐标点
X = rdm.randn(300, 2)
# 两种颜色的点
Y_ = [int(x0 * x0 + x1 * x1 < 2) for (x0, x1) in X]

Y_color = [['red' if y else 'blue'] for y in Y_]

X = np.vstack(X).reshape(-1, 2)
Y_ = np.vstack(Y_).reshape(-1, 1)

print(X)
print(Y_)
print(Y_color)

'''
X 是一个 300 行 2 列的张量 2列是一个 点的坐标
Y_ 是一个 300 行 1 列的张量 经过计算后 x1的平方 + x2的平方的和 小于 2 是1 其他是 0
Y_color 是一个 包含 red 和 blue 的数组
'''

# x坐标集 y坐标集 c 颜色
plt.scatter(X[:, 0], X[:, 1], c=np.squeeze(Y_color))
plt.show()


# 获取w方法
def get_weight(shape, regularizer):
    w = tf.Variable(tf.random_normal(shape), dtype=tf.float32)
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w


# 获取b方法
def get_bias(shape):
    b = tf.Variable(tf.constant(0.01, shape=shape))
    return b


# 输入数据集 参数 向前传播
x = tf.placeholder(tf.float32, shape=(None, 2))
y_ = tf.placeholder(tf.float32, shape=(None, 1))

w1 = get_weight([2, 11], 0.01)
b1 = get_bias([11])
y1 = tf.nn.relu(tf.matmul(x, w1) + b1)

w2 = get_weight([11, 1], 0.01)
b2 = get_bias([1])
y = tf.matmul(y1, w2) + b2  # 输出层 不激活

# 定义损失函数

loss_mse = tf.reduce_mean(tf.square(y - y_))
# 加上每一个 正则化 W的损失
loss_total = loss_mse + tf.add_n(tf.get_collection('losses'))

# 反向传播方式 不包括正则化
# train_step = tf.train.AdamOptimizer(0.0001).minimize(loss_mse)
# train_step = tf.train.GradientDescentOptimizer(0.0001).minimize(loss_mse)
# 反向传播方式 包括正则化
train_step = tf.train.AdamOptimizer(0.0001).minimize(loss_total)

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    STEPS = 40000
    for i in range(STEPS):
        start = (i * BATCH_SIZE) % 300
        end = start + BATCH_SIZE
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y_[start:end]})
        if i % 2000 == 0:
            # loss_mse_v = sess.run(loss_mse, feed_dict={x: X, y_: Y_})
            loss_mse_v = sess.run(loss_total, feed_dict={x: X, y_: Y_})
            print("After %d steps, loss is: %f" % (i, loss_mse_v))

    xx, yy = np.mgrid[-3:3:.01, -3:3:.01]
    grid = np.c_[xx.ravel(), yy.ravel()]

    probs = sess.run(y, feed_dict={x: grid})
    print(' 验证测试 ')
    print(probs)
    probs = probs.reshape(xx.shape)
    print(' reshape ')
    print(probs)

plt.scatter(X[:, 0], X[:, 1], c=np.squeeze(Y_color))
plt.contour(xx, yy, probs, levels=[.5])
plt.show()
