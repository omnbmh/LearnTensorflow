# -8- coding:utf-8 -8-

'''
优化 learning_rate 参数

'''

import tensorflow as tf

# 定义 待优化的参数w 初始为 5
w = tf.Variable(tf.constant(5, dtype=tf.float32))

# 定义损失函数 loss w+1 的平方

loss = tf.square(w + 1)

# 定义反向传播过程
# train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

'''
学习率大了不收敛 学习率小了收敛速度慢 
学习率设置多少合适呢？
引入指数衰减学习率
'''
LEARNING_RATE_BASE = 0.2
LEARNING_RATE_DECAY = 0.99  # 越小衰减越快
LEARNING_RATE_STEP = 1  # 喂入多少轮数据后 更新学习率 一般设置为 总样本数/BATCH_SIZE
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,
                                           global_step,
                                           LEARNING_RATE_STEP,
                                           LEARNING_RATE_DECAY,
                                           staircase=True)

train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

# 训练 50 轮
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    for i in range(50):
        sess.run(train_step)
        # w_val, loss_val = sess.run([w, loss])
        # print("After %d training steps w is %f loss is %f." % (i, w_val, loss_val))
        w_val, loss_val, step, learning_rate_val = sess.run([w, loss, global_step, learning_rate])
        print("After %d training steps gloal_step is %f w is %f learning rate is %f loss is %f." % (
            i, step, w_val, learning_rate_val, loss_val))

'''
learning_rate is 0.2  31 step 

'''
