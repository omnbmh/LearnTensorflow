# -8- coding:utf-8 -8-

import tensorflow as tf

a = tf.constant([1.0, 2.0])
b = tf.constant([3.0, 4.0])

result = a + b
print(result)  # 是一张计算图 并没有实际计算

y = tf.matmul(a, b)
print(y)

# output
# Tensor("add:0", shape=(2,), dtype=float32)
# add:0 节点名:第0个输出  shape=(2,) 维度 一维数组 长度2 float32 数据类型

# 上面输出的是计算图 并不计算
# 要想计算出值 我们会用到会话

with tf.Session() as sess:
    print(sess.run(result))
