# -8- coding:utf-8 -8-

'''
制作数据集
'''

import tensorflow as tf
import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import os
import random

image_train_path = './mnist_data_jpg/mnist_train_jpg_10000'
label_train_path = './mnist_data_jpg/mnist_train_jpg_10000.txt'
tfrecord_train = './data/mnist_train.tfrecords'
image_test_path = './mnist_data_jpg/mnist_test_jpg_1000'
label_test_path = './mnist_data_jpg/mnist_test_jpg_1000.txt'
tfrecord_test = './data/mnist_test.tfrecords'
data_path = './data'
resize_height = 28
resize_width = 28


def write_tfrecord(tfrecord_name, image_path, label_path):
    writer = tf.python_io.TFRecordWriter(tfrecord_name)
    num_pic = 0
    contents = []
    with open(label_path, 'r') as f:
        contents = f.readlines()

    for content in contents:
        value = content.split()
        print(str(value[0]) + " : " + str(value[1]))
        img_path = os.path.join(image_path, value[0])
        img = Image.open(img_path)
        img_raw = img.tobytes()
        labels = [0] * 10
        labels[int(value[1])] = 1
        example = tf.train.Example(features=tf.train.Features(feature={
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=labels))
        }))
        writer.write(example.SerializeToString())
        num_pic += 1
        print("the number of picture: ", num_pic)
    writer.close()
    print("write tfrecord successful")


def gen_tfrecord():
    if os.path.exists(data_path):
        print("The directory already exists")
    else:
        os.makedirs(data_path)
        print("The directory was created successfully")

    write_tfrecord(tfrecord_train, image_train_path, label_train_path)
    write_tfrecord(tfrecord_test, image_test_path, label_test_path)


def read_tfrecord(tfrecord_path):
    filename_queue = tf.train.string_input_producer([tfrecord_path])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([10], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string)
                                       })
    img = tf.decode_raw(features["img_raw"], tf.uint8)
    img.set_shape([784])
    img = tf.cast(img, tf.float32) * (1. / 255)
    label = tf.cast(features['label'], tf.float32)
    return img, label


def get_tfrecord(num, is_train=False):
    if is_train:
        tfrecord_path = tfrecord_train
    else:
        tfrecord_path = tfrecord_test
    img, label = read_tfrecord(tfrecord_path)
    img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                    batch_size=num,
                                                    num_threads=2,
                                                    capacity=1000,
                                                    min_after_dequeue=700)
    return img_batch, label_batch


def gen_image():
    '''
    生成图片 用于组装测试集
    :return:
    '''
    with open(label_train_path, 'a+') as train_txt:
        # for train 10000
        for i in range(10000):
            num = str(random.randint(0, 9))
            filename = str(i).zfill(4) + '.png'
            generate_captcha(image_train_path, filename, num)
            train_txt.write(filename + ' ' + num + '\n')

    with open(label_test_path, 'a+') as test_txt:
        # for test 1000
        for i in range(1000):
            num = str(random.randint(0, 9))
            filename = str(i).zfill(4) + '.png'
            generate_captcha(image_test_path, filename, num)
            test_txt.write(filename + ' ' + num + '\n')


def generate_captcha(image_path, filename, num):
    # 生成一个Image 指定大小 和 背景色
    image = Image.new('RGB', (36, 36), (255, 255, 255))

    # 来一个画笔
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("fonts/msyh.ttf", size=32)

    # 把字写到Image上
    draw.text((6, 0), str(num), (0, 0, 0), font=font)

    image.save(os.path.join(image_path, filename), 'png')


if __name__ == '__main__':
    # gen_image()
    # gen_tfrecord()
    img_batch, label_batch = get_tfrecord(100, False)
    print(img_batch)
    print(label_batch)
    '''
    # 读出来了
    Tensor("shuffle_batch:0", shape=(100, 784), dtype=float32)
    Tensor("shuffle_batch:1", shape=(100, 10), dtype=float32)
    '''
