# -8- coding:utf-8 -8-

# 两层简单神经网络(全连接)
import tensorflow as tf

# 定义输入和参数

# x = tf.constant([[0.7, 0.5]])

# 使用占位符 喂入
# x = tf.placeholder(tf.float32, shape=(None, 2))
# w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
# w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

# 定义前向传播过程

# a = tf.matmul(x, w1)
# y = tf.matmul(a, w2)

# 计算结果
# with tf.Session() as sess:
#     init_op = tf.global_variables_initializer()
#     sess.run(init_op)
#     # print (sess.run(y))
#     print (sess.run(y, feed_dict={x: [[0.7, 0.5], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5]]}))

# 加上反向传播 来真正模拟一个训练过程
import numpy as np

# 生成训练数据 和 验证集
BATCH_SIZE = 8
seed = 23455

# 32行 2 列  32个 包含 2个特征的数据 作为数据集
rng = np.random.RandomState(seed)
X = rng.rand(32, 2)

# 人为的制造一套标准  来验证结果 特征0 加 特征1 小于1 人为是好的
Y = [[int(x0 + x1 < 1)] for (x0, x1) in X]

print (X)
print (Y)

# 定义神经网络的输入

x = tf.placeholder(tf.float32, shape=(None, 2))
y_ = tf.placeholder(tf.float32, shape=(None, 1))  # 标准答案

w1 = tf.Variable(tf.random_normal((2, 3), stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal((3, 1), stddev=1, seed=1))
# 定义前向传播过程

a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

# 定义 损失函数 及 反向传播方法
loss = tf.reduce_mean(tf.square(y - y_))
# train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
# train_step = tf.train.MomentumOptimizer(0.001,0.9).minimize(loss)
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)
# 生成会话 训练
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    print("w1 ---")
    print (sess.run(w1))
    print("w2 ---")
    print (sess.run(w2))
    STEPS = 10000
    for i in range(STEPS):
        start = (i * BATCH_SIZE) % 32
        end = start + BATCH_SIZE
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start: end]})
        if i % 500 == 0:
            total_loss = sess.run(loss, feed_dict={x: X, y_: Y})
            print ("After %d training steps,loss on all data is %g" % (i, total_loss))

    # 输出训练后的值
    print("w1 ---")
    print (sess.run(w1))
    print("w2 ---")
    print (sess.run(w2))
