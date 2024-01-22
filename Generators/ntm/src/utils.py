import numpy as np
import tensorflow as tf
import traceback

def expand(x, dim, N):
    for line in traceback.format_stack():
        print(line.strip())

#     print("Expand")
#     print("x:", x)
#     print("dim:", dim)
#     print("N:",N)
#     tmp = tf.concat([tf.expand_dims(x, dim) for _ in range(N)], axis=dim)
#     print("tf.concat([tf.expand_dims(x, dim) for _ in range(N)], axis=dim):", tmp)
#     with tf.Session() as sess:
#         print(tmp.eval())

    return tf.concat([tf.expand_dims(x, dim) for _ in range(N)], axis=dim)

def learned_init(units):
    return tf.squeeze(tf.contrib.layers.fully_connected(tf.ones([1, 1]), units,
        activation_fn=None, biases_initializer=None))

def create_linear_initializer(input_size, dtype=tf.float32):
    stddev = 1.0 / np.sqrt(input_size)
    return tf.truncated_normal_initializer(stddev=stddev, dtype=dtype)