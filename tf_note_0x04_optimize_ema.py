# -8- coding:utf-8 -8-

'''
优化 ema 滑动平均 参数
记录了每个参数一段时间内过往值的平均，增加了模型的泛化性

'''
import tensorflow as tf

MOVING_AVERAGE_DECAY = 0.99  # 值越大 追随的越慢

w1 = tf.Variable(0, dtype=tf.float32)
global_step = tf.Variable(0, trainable=False)
ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)

# ema_op = ema.apply([w1]) # 指定记录滑动平均的参数
ema_op = ema.apply(tf.trainable_variables())  # 记录所有参数的滑动平均

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    # 打印出 w1 和滑动平均
    print(sess.run([w1, ema.average(w1)]))

    # 将 w1 赋值为 1
    sess.run(tf.assign(w1, 1))
    sess.run(ema_op)
    print(sess.run([w1, ema.average(w1)]))

    # 模拟出 100 轮迭代后 w1 10
    sess.run(tf.assign(global_step, 100))
    sess.run(tf.assign(w1, 10))
    sess.run(ema_op)
    print(sess.run([w1, ema.average(w1)]))

    # 每运行一次 就会更新一次滑动平均值
    sess.run(ema_op)
    print(sess.run([w1, ema.average(w1)]))

    sess.run(ema_op)
    print(sess.run([w1, ema.average(w1)]))

    sess.run(ema_op)
    print(sess.run([w1, ema.average(w1)]))

    sess.run(ema_op)
    print(sess.run([w1, ema.average(w1)]))

    sess.run(ema_op)
    print(sess.run([w1, ema.average(w1)]))
