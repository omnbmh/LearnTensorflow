# LearnTensorflow
Tensorflow学习

从零开始学习

### 验证环境 `helloworld.py`
```python
import tensorflow as tf

hw = tf.constant("Hello Tensotflow!")

with tf.Session() as sess:
    print(sess.run(hw))

==========
Hello Tensotflow!

```