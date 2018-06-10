import tensorflow as tf


def concat(tensors, axis, *args, **kwargs):
    return tf.concat_v2(tensors, axis, *args, **kwargs) if "concat_v2" in dir(tf)\
        else tf.concat(tensors, axis, *args, **kwargs)


def conv_cond_concat(x, y):
    """Concatenate conditioning vector as additional feature maps."""
    x_shape = tf.shape(x)
    y_shape = tf.shape(y)
    return concat([x, y*tf.ones([x_shape[0], y_shape[1], x_shape[2], x_shape[3]])], 1)