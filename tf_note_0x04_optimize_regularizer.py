# -8- coding:utf-8 -8-
'''
使用正则化 w 优化模型

数据 [x1,x2] 为正态分布的随机点坐标
当 Y0 = x2+y2 < 2 时 为 red 其余 为 blue

'''
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 每次喂入神经网路的样本数
BATCH_SIZE = 30
seed = 2
rdm = np.random.RandomState(seed)
# 随机生成 300 个坐标点
X = rdm.randn(300, 2)
# 两种颜色的点 x0的平方 + x1的平方 小于 2 存入 1 否则 是 0
Y_ = [int(x0 * x0 + x1 * x1 < 2) for (x0, x1) in X]
# 1 转 red ; 0 转 blue
Y_color = [['red' if y else 'blue'] for y in Y_]

X = np.vstack(X).reshape(-1, 2)  # 整理成 n行2列
Y_ = np.vstack(Y_).reshape(-1, 1)  # 整理成 n行1列

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

# 损失函数
loss_mse = tf.reduce_mean(tf.square(y - y_))
# 加上每一个 正则化 W的损失
loss_total = loss_mse + tf.add_n(tf.get_collection('losses'))

# 反向传播方式 不包括正则化
# train_step = tf.train.AdamOptimizer(0.0001).minimize(loss_mse)
# train_step = tf.train.GradientDescentOptimizer(0.0001).minimize(loss_mse)

# 反向传播方式 包括正则化
train_step = tf.train.AdamOptimizer(0.0001).minimize(loss_mse)

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    STEPS = 40000
    for i in range(STEPS):
        start = (i * BATCH_SIZE) % 300
        end = start + BATCH_SIZE
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y_[start:end]})
        if i % 2000 == 0:
            loss_mse_v = sess.run(loss_mse, feed_dict={x: X, y_: Y_})
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
# 画一条 点 高度的线 将 yy值为0.5 的上色
plt.contour(xx, yy, probs, levels=[.5])
plt.show()

'''
After 0 steps, loss is: 7.637332
After 2000 steps, loss is: 2.148881
After 4000 steps, loss is: 1.025571
After 6000 steps, loss is: 0.495696
After 8000 steps, loss is: 0.243670
After 10000 steps, loss is: 0.149529
After 12000 steps, loss is: 0.112888
After 14000 steps, loss is: 0.096693
After 16000 steps, loss is: 0.088393
After 18000 steps, loss is: 0.083354
After 20000 steps, loss is: 0.080420
After 22000 steps, loss is: 0.078421
After 24000 steps, loss is: 0.076730
After 26000 steps, loss is: 0.075607
After 28000 steps, loss is: 0.074844
After 30000 steps, loss is: 0.074232
After 32000 steps, loss is: 0.073802
After 34000 steps, loss is: 0.073541
After 36000 steps, loss is: 0.073345
After 38000 steps, loss is: 0.073224
'''

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
            loss_total_v = sess.run(loss_total, feed_dict={x: X, y_: Y_})
            print("After %d steps, loss is: %f" % (i, loss_total_v))

    xx, yy = np.mgrid[-3:3:.01, -3:3:.01]
    grid = np.c_[xx.ravel(), yy.ravel()]

    probs = sess.run(y, feed_dict={x: grid})
    print(' 验证测试 ')
    print(probs)
    probs = probs.reshape(xx.shape)
    print(' reshape ')
    print(probs)

plt.scatter(X[:, 0], X[:, 1], c=np.squeeze(Y_color))
# 画一条 点 高度的线 将 yy值为0.5 的上色
plt.contour(xx, yy, probs, levels=[.5])
plt.show()

'''
After 0 steps, loss is: 3.575430
After 2000 steps, loss is: 0.557260
After 4000 steps, loss is: 0.277018
After 6000 steps, loss is: 0.230276
After 8000 steps, loss is: 0.204616
After 10000 steps, loss is: 0.186064
After 12000 steps, loss is: 0.172204
After 14000 steps, loss is: 0.159668
After 16000 steps, loss is: 0.149617
After 18000 steps, loss is: 0.141936
After 20000 steps, loss is: 0.135359
After 22000 steps, loss is: 0.129584
After 24000 steps, loss is: 0.124448
After 26000 steps, loss is: 0.120160
After 28000 steps, loss is: 0.116935
After 30000 steps, loss is: 0.114290
After 32000 steps, loss is: 0.111941
After 34000 steps, loss is: 0.109829
After 36000 steps, loss is: 0.107926
After 38000 steps, loss is: 0.106093
'''