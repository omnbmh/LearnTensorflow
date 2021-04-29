# LearnTensorflow
Tensorflow学习

从零开始学习

Python 3.7
pip 指定 下载阿里云源

```
pip install --index-url http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com tensorflow==1.15.3
```

### 验证环境 `helloworld.py`
```python
import tensorflow as tf

hw = tf.constant("Hello Tensotflow!")

with tf.Session() as sess:
    print(sess.run(hw))

==========
Hello Tensotflow!

```