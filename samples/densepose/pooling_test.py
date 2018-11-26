import tensorflow as tf
import numpy as np

sess = tf.Session()
predicted_mask = tf.random_uniform(shape=(200, 25, 56, 56))
target_c_i = tf.random_uniform(shape=(200, 5, 196), minval=0, maxval=55)
predicted_mask = tf.argmax(predicted_mask, axis=1)
print(sess.run([target_c_i, predicted_mask]))

target_x = tf.cast(target_c_i[0, 1, :], tf.int32)
target_y = tf.cast(target_c_i[1, 1, :], tf.int32)
sample_pos = tf.fill((56, 56), -1)
coords = tf.transpose(tf.stack([target_x, target_y]))
# get tensor with 1's at positions (row1, col1),...
binary_mask = tf.sparse_to_dense(coords, tf.shape(sample_pos), 1)
sess.run([target_x, target_y, sample_pos, coords, binary_mask])
# convert 1/0 to True/False
binary_mask = tf.cast(binary_mask, tf.bool)

tf.where(binary_mask, tf.cast(predicted_mask[0, :, :], tf.int32), sample_pos)



print(sess.run(predicted_mask))