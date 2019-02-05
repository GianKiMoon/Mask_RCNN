import keras.backend as K
import keras
import tensorflow as tf
from keras.utils import to_categorical


def m_loss(x_true, x_pred):
    return K.mean(K.categorical_crossentropy(x_true, x_pred))


def i_loss(i_true, i_pred):
    # # Flat tensors
    # i_true = tf.reshape(i_true, [-1])
    # i_pred = tf.reshape(i_pred, [-1])
    #
    # # Filter not defined values
    # idx = tf.where(tf.not_equal(i_true, tf.constant(-1.0)))
    # i_true = tf.gather(i_true, idx)
    # i_pred = tf.gather(i_pred, idx)

    return K.mean(K.categorical_crossentropy(i_true, i_pred))


def smooth_l1_loss(y_true, y_pred):
    """Implements Smooth-L1 loss.
    y_true and y_pred are typically: [N, 4], but could be any shape.
    """
    diff = K.abs(y_true - y_pred)
    less_than_one = K.cast(K.less(diff, 1.0), "float32")
    loss = (less_than_one * 0.5 * diff**2) + (1 - less_than_one) * (diff - 0.5)
    return loss


def uv_loss(uv_true, i_true, uv_pred):
    uv_true = tf.boolean_mask(uv_true, i_true)
    uv_pred = tf.boolean_mask(uv_pred, i_true)
    uv_true = tf.reshape(uv_true, [-1])
    uv_pred = tf.reshape(uv_pred, [-1])

    idx = tf.where(tf.not_equal(uv_true, tf.constant(-1.0)))
    uv_true = tf.gather(uv_true, idx)
    uv_pred = tf.gather(uv_pred, idx)

    loss = smooth_l1_loss(uv_true, uv_pred)
    loss = K.switch(tf.size(loss) > 0, K.mean(loss), tf.constant(0.0))

    return loss

def gps_ap(iuv_true, iuv_pred):
    return None












# # Reshape for simplicity. Merge first two dimensions into one.
#     target_c_i = i_true
#     predicted_c_i = i_pred
#     target_shape = tf.shape(target_c_i)
#     pred_shape = tf.shape(predicted_c_i)
#
#     print("t shape", target_c_i)
#     print("p shape", predicted_c_i)
#
#     # Permute predicted masks to [N, num_classes, height, width]
#     predicted_c_i = tf.transpose(predicted_c_i, [0, 3, 1, 2])
#     # Reduce to indeces of max values
#     predicted_c_i = tf.argmax(predicted_c_i, axis=1)
#
#     def pool_slice(x):
#         # Get target and prediction slice
#         t_slice = x[0]
#         p_slice = x[1]
#
#         # Extract ground truth x and y coordinates from target slice
#         target_x = tf.cast(t_slice[0, :], tf.int32)
#         target_y = tf.cast(t_slice[1, :], tf.int32)
#
#         # Remove negative entries
#         target_x = tf.reshape(tf.gather(target_x, tf.where(target_x > -1)), [-1])
#         target_y = tf.reshape(tf.gather(target_y, tf.where(target_y > -1)), [-1])
#
#         def do_pool():
#             sample_pos = tf.fill((pred_shape[2], pred_shape[2]), -1)
#
#             t_x = tf.cast(tf.reshape(target_x, [-1]), tf.int32)
#             t_y = tf.cast(tf.reshape(target_y, [-1]), tf.int32)
#
#             # Format coords and filter out duplicates
#             coords = tf.transpose(tf.stack([t_x, t_y]))
#             print("coords ", coords)
#
#             # Get the unique coordinates and find index of duplicate ones
#             unique_coords, unique_coords_idx = tf_unique_2d(coords)
#
#             t_slice_2 = tf.cast(t_slice[2, :], tf.int32)
#
#             # Get ground truth slice gathered by unique coords and without negative entries
#
#             # t_slice_2 = tf.reshape(tf.gather(t_slice_2, tf.where(target_x > -1)), [-1])
#             t_slice_2_unique = tf.gather(t_slice_2, unique_coords_idx)
#
#             # Create binary mask of coords
#             max_coord = tf.reduce_max(unique_coords)
#             binary_mask = tf.sparse_to_dense(unique_coords, tf.shape(sample_pos), 1, validate_indices=False)
#             binary_mask = tf.cast(binary_mask, tf.bool)
#
#             # Analogue mask for point indeces
#             coord_ix = tf.range(tf.shape(unique_coords)[0])
#             binary_mask_idx = tf.sparse_to_dense(unique_coords, tf.shape(sample_pos), coord_ix,
#                                                  default_value=-1, validate_indices=False)
#
#             # Reshape masks
#             binary_mask = tf.reshape(binary_mask, [-1])
#             binary_mask_idx = tf.reshape(binary_mask_idx, [-1])
#
#             # Pool the actual prediction (do the same to the indeces
#             p_slice_flat = tf.reshape(p_slice, [-1])
#             pooled_idx = tf.boolean_mask(binary_mask_idx, binary_mask)
#             pooled_vals = tf.boolean_mask(p_slice_flat, binary_mask)
#
#             # Sort pooled indexes (descending) and reverse: now the pooled values are arranged compatible to the gt
#             pooled_vals_sorted_reversed = tf.gather(pooled_vals, tf.nn.top_k(pooled_idx,
#                                                                              k=tf.shape(pooled_idx)[0]).indices)
#             pooled_vals_sorted = tf.reverse(pooled_vals_sorted_reversed, [-1])
#
#             t = 196 - tf.shape(pooled_vals_sorted)[0]
#             paddings = tf.concat(([[0, 0]], [[0, t]]), axis=0)
#
#             pooled_vals_sorted = tf.reshape(pooled_vals_sorted, [1, -1])
#             pooled_vals_sorted = tf.pad(pooled_vals_sorted, paddings, 'CONSTANT', constant_values=-1)
#             pooled_vals_sorted = tf.reshape(pooled_vals_sorted, [-1])
#
#             t_slice_2_unique = tf.reshape(t_slice_2_unique, [1, -1])
#             t_slice_2_unique = tf.pad(t_slice_2_unique, paddings, 'CONSTANT', constant_values=-1)
#             t_slice_2_unique = tf.reshape(t_slice_2_unique, [-1])
#
#             #p = tf.Print([t_slice_new.get_shape()], t_slice_new, "Tslice new")
#
#             return tf.cast(t_slice_2_unique, tf.int32), tf.cast(pooled_vals_sorted, tf.int32)
#
#         t_slice_new, p_slice_new = tf.cond(tf.greater(tf.shape(target_x)[0], tf.zeros(shape=(), dtype=tf.int32)),
#                                            do_pool,
#                                            lambda: (tf.zeros([196], dtype=tf.int32), tf.zeros([196], dtype=tf.int32))
#                                            )
#         p_slice_new = tf.reshape(p_slice_new, [1, 196])
#         t_slice_new = tf.reshape(t_slice_new, [1, 196])
#         p_slice_new = tf.cast(p_slice_new, tf.float32)
#         t_slice_new = tf.cast(t_slice_new, tf.float32)
#
#         return t_slice_new, p_slice_new
#
#     target_c_i = tf.reshape(target_c_i, [-1, 5, 196])
#     predicted_c_i = tf.reshape(predicted_c_i, [-1, pred_shape[2], pred_shape[2]])
#
#     (target_c_i, predicted_c_i) = tf.map_fn(pool_slice,
#                                             (target_c_i, predicted_c_i),
#                                             dtype=(tf.float32, tf.float32),
#                                             infer_shape=True)
#
#     target_c_i = tf.cast(target_c_i, tf.int32)
#     predicted_c_i = tf.cast(predicted_c_i, tf.int32)
#
#     loss = keras.losses.categorical_crossentropy(tf.one_hot(target_c_i, 25), tf.one_hot(predicted_c_i, 25))
#     loss = K.mean(loss)
#
#
# def tf_unique_2d(x):
#     x_shape = tf.shape(x)  # (3,2)
#
#     x1 = tf.tile(x, [1, x_shape[0]])  # [[1,2],[1,2],[1,2],[3,4],[3,4],[3,4]..]
#     x2 = tf.tile(x, [x_shape[0], 1])  # [[1,2],[1,2],[1,2],[3,4],[3,4],[3,4]..]
#
#     x1_2 = tf.reshape(x1, [x_shape[0] * x_shape[0], x_shape[1]])
#     x2_2 = tf.reshape(x2, [x_shape[0] * x_shape[0], x_shape[1]])
#     cond = tf.reduce_all(tf.equal(x1_2, x2_2), axis=1)
#     cond = tf.reshape(cond, [x_shape[0], x_shape[0]])  # reshaping cond to match x1_2 & x2_2
#     cond_shape = tf.shape(cond)
#     cond_cast = tf.cast(cond, tf.int32)  # convertin condition boolean to int
#     cond_zeros = tf.zeros(cond_shape, tf.int32)  # replicating condition tensor into all 0's
#
#     # CREATING RANGE TENSOR
#     r = tf.range(x_shape[0])
#     r = tf.add(tf.tile(r, [x_shape[0]]), 1)
#     r = tf.reshape(r, [x_shape[0], x_shape[0]])
#
#     # converting TRUE=1 FALSE=MAX(index)+1 (which is invalid by default) so
#     # when we take min it wont get selected & in end we will only take values <max(indx).
#     f1 = tf.multiply(tf.ones(cond_shape, tf.int32), x_shape[0] + 1)
#     f2 = tf.ones(cond_shape, tf.int32)
#     cond_cast2 = tf.where(tf.equal(cond_cast, cond_zeros), f1, f2)  # if false make it max_index+1 else keep it 1
#
#     # multiply range with new int boolean mask
#     r_cond_mul = tf.multiply(r, cond_cast2)
#     r_cond_mul2 = tf.reduce_min(r_cond_mul, axis=1)
#     r_cond_mul3, unique_idx = tf.unique(r_cond_mul2)
#     r_cond_mul4 = tf.subtract(r_cond_mul3, 1)
#
#     # get actual values from unique indexes
#     op = tf.gather(x, r_cond_mul4)
#
#     return op, r_cond_mul4
