# -8- coding:utf-8 -8-
'''
实现手写数字图片的识别

'''
import tensorflow as tf
import numpy as np
import os
import time
from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image

INPUT_NODE = 28 * 28  # 784
OUTPUT_NODE = 10
LAYER1_NODE = 500


def get_weight(shape, regularizer):
    w = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    if regularizer != None:
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w


def get_bias(shape):
    b = tf.Variable(tf.zeros(shape))
    return b


# 神经网络结构
def forward(x, regularizer):
    w1 = get_weight([INPUT_NODE, LAYER1_NODE], regularizer)
    b1 = get_bias([LAYER1_NODE])
    y1 = tf.nn.relu(tf.matmul(x, w1) + b1)

    w2 = get_weight([LAYER1_NODE, OUTPUT_NODE], regularizer)
    b2 = get_bias([OUTPUT_NODE])
    y = tf.matmul(y1, w2) + b2  # 输出层 不用激活
    return y


STEPS = 10000  # 训练多少次
BATCH_SIZE = 200  # 每次训练样本数
LEARNING_RATE_BASE = 0.1  # 基础学习速率
LEARNING_RATE_DECAY = 0.99  # 学习率衰减率
REGULARIZER = 0.0001  # 正则化
MOVING_AVERAGE_DECAY = 0.99  # 滑动平均衰减率
MODEL_SAVE_PATH = './model/'
MODEL_NAME = 'mnist_model'


def backward(mnist):
    x = tf.placeholder(tf.float32, shape=(None, INPUT_NODE))
    y_ = tf.placeholder(tf.float32, shape=(None, OUTPUT_NODE))
    y = forward(x, REGULARIZER)
    global_step = tf.Variable(0, trainable=False)

    # 交叉熵
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cem = tf.reduce_mean(ce)
    loss = cem + tf.add_n(tf.get_collection('losses'))

    # 定义指数衰减学习率
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE,
        LEARNING_RATE_DECAY,
        staircase=True)

    # 定义反向传播方法 包含正则化
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    ema_op = ema.apply(tf.trainable_variables())

    with tf.control_dependencies([train_step, ema_op]):
        train_op = tf.no_op(name="train")

    saver = tf.train.Saver()

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        # 断点续训
        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        for i in range(STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_v, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
            if i % 1000 == 0:
                print("After %d training steps, loss is %g" % (step, loss_v))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)


def test(mnist):
    x = tf.placeholder(tf.float32, shape=(None, INPUT_NODE))
    y_ = tf.placeholder(tf.float32, shape=(None, OUTPUT_NODE))
    y = forward(x, None)
    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
    ema_restore = ema.variables_to_restore()
    saver = tf.train.Saver(ema_restore)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    while True:
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                print(global_step)
                accuracy_score = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
                print("After %s training step, test accuracy = %g" % (global_step, accuracy_score))
            else:
                print("No checkpoint file found!")
                return
        time.sleep(5)


# Application
def restore_model(test_pic_arr):
    with tf.Graph().as_default() as tg:
        x = tf.placeholder(tf.float32, [None, INPUT_NODE])
        y = forward(x, None)
        pre_value = tf.argmax(y, 1)

        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
        variable_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variable_to_restore)

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                pre_value = sess.run(pre_value, feed_dict={x: test_pic_arr})
                return pre_value
            else:
                print("No checkpoint file found!")
                return -1


def pre_pic(pic_name):
    img = Image.open(pic_name)
    re_img = img.resize((28, 28), Image.ANTIALIAS)  # 消除锯齿 resize
    img_arr = np.array(re_img.convert('L'))  # 灰度图
    # 反色
    threshold = 50
    for i in range(28):
        for j in range(28):
            img_arr[i][j] = 255 - img_arr[i][j]
            if img_arr[i][j] < threshold:
                img_arr[i][j] = 0
            else:
                img_arr[i][j] = 255

    # 将处理过的图片 保存一下
    Image.fromarray(img_arr).save("cvt_" + pic_name)

    nm_arr = img_arr.reshape([1, 28 * 28])
    nm_arr = nm_arr.astype(np.float32)
    img_ready = np.multiply(nm_arr, 1.0 / 255.0)
    return img_ready


def application():
    test_num = input("input the number of test pictures:")
    for i in range(test_num):
        test_pic = raw_input("the path of test picture:")
        test_pic_arr = pre_pic(test_pic)
        pre_value = restore_model(test_pic_arr)
        print("The prediction number is : ")
        print(pre_value)


def main():
    mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)
    # backward(mnist)
    # test(mnist)
    application()


if __name__ == '__main__':
    main()
