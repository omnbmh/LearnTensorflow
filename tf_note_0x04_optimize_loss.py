# -8- coding:utf-8 -8-
'''
优化 loss 参数

使用三种优化方法 均方误差 自定义 交叉熵

预测 可乐的销量 影响特征 价格 x1 包装 x2
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
# 人为的制造一套标准  来验证结果 两个特征相加 正负0.05 噪声
Y = [[x0 + x1 + (rng.rand() / 10.0 - 0.05)] for (x0, x1) in X]

# 定义神经网络的输入 输出
x = tf.placeholder(tf.float32, shape=(None, 2))
y_ = tf.placeholder(tf.float32, shape=(None, 1))

# 定义前向传播
w1 = tf.Variable(tf.random_normal((2, 1), stddev=1, seed=1))
y = tf.matmul(x, w1)

# 定义 损失函数 及 反向传播方法

# mse 均值方差
# loss = tf.reduce_mean(tf.square(y_ - y))
# 自定义损失函数
# 实际值 大于 预测值 COST 否则 PROFIT
# loss = tf.reduce_sum(tf.where(tf.greater(y, y_), (y - y_), 9 * (y_ - y)))

# 交叉熵 loss
ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
loss = tf.reduce_mean(ce)

# 反向传播方法为梯度下降
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
# 生成会话 训练
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    STEPS = 20000
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
    print("Final w1 ---")
    print (sess.run(w1))

'''
Final w1 --- 均值方差
[[0.98019385]
 [1.0159807 ]]
 
 Final w1 --- 自定义
[[1.020171 ]
 [1.0425103]]
 
 Final w1 --- 交叉熵
[[-0.8113182]
 [ 1.4845988]]
'''
