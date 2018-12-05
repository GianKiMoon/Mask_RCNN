import tensorflow as tf
import numpy as np

sess = tf.Session()

a = tf.constant([2, 4, 1, 6, 3, 6, 7, 2, 5, 24])
a = tf.one_hot(indices=a, depth=25)

print(sess.run([a]))