import tensorflow as tf
import numpy as np

sess = tf.Session()

w = 3

a = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
b = tf.reshape(a, [-1])
print(sess.run([a, b]))