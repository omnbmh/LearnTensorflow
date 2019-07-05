# -8- coding:utf-8 -8-
'''
验证环境是否正常
'''
import tensorflow as tf

hw = tf.constant("Hello Tensotflow!")

with tf.Session() as sess:
    print(sess.run(hw))
