# -8- coding:utf-8 -8-
'''
预测 可乐的销量 影响特征 x1 x2
'''
# 两层简单神经网络(全连接)
import tensorflow as tf
import numpy as np

# 生成训练数据 和 验证集
BATCH_SIZE = 8
SEED = 23455

# 32行 2 列  32个 包含 2个特征的数据 作为数据集
rng = np.random.RandomState(SEED)
X = rng.rand(32, 2)
# 人为的制造一套标准  来验证结果
Y = [[x0 + x1 + (rng.rand() / 10.0 - 0.05)] for (x0, x1) in X]

# 定义神经网络的输入
x = tf.placeholder(tf.float32, shape=(None, 2))
y_ = tf.placeholder(tf.float32, shape=(None, 1))
w1 = tf.Variable(tf.random_normal((2, 1), stddev=1, seed=1))
y = tf.matmul(x, w1)

# 定义 损失函数 及 反向传播方法
loss = tf.reduce_mean(tf.square(y_ - y))
# 自定义损失函数
loss = tf.reduce_sum(tf.where(tf.greater(y, y_), 9*(y - y_), (y_ - y)))
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
# 生成会话 训练
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    STEPS = 2000
    for i in range(STEPS):
        start = (i * BATCH_SIZE) % 32
        end = start + BATCH_SIZE
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start: end]})
        if i % 500 == 0:
            total_loss = sess.run(loss, feed_dict={x: X, y_: Y})
            print ("After %d training steps,loss on all data is %g" % (i, total_loss))
            print ("After %d training steps,w1 is " % i)
            print (sess.run(w1))

    # 输出训练后的值
    print("w1 ---")
    print (sess.run(w1))
