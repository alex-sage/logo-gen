import math
import numpy as np
import tensorflow as tf

from tensorflow.python.framework import ops

# from utils import *

# Note: omitted backwards compatibility in order to avoid warnings
image_summary = tf.summary.image
scalar_summary = tf.summary.scalar
histogram_summary = tf.summary.histogram
merge_summary = tf.summary.merge
SummaryWriter = tf.summary.FileWriter

class batch_norm(object):
    def __init__(self, epsilon=1e-5, momentum=0.9, name='batch_norm'):
        self.epsilon = epsilon
        self.momentum = momentum
        self.name = name

    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(x,
                                            decay=self.momentum,
                                            updates_collections=None,
                                            epsilon=self.epsilon,
                                            scale=True,
                                            is_training=train,
                                            scope=self.name)


def conv2d(input_, output_dim,
           k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
           name='conv2d'):
    with tf.variable_scope(name):
        # input shape: [batch, in_height, in_width, in_channels]
        # filter shape : [height, width, input_channels, output_channels]
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')
        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0))
        conv = tf.nn.bias_add(conv, biases)
        # output shape: [batch, out_height, out_width, output_channels]
        return conv


def deconv2d(input_, output_shape,
             k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
             name="deconv2d", with_w=False):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        _deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape, strides=[1, d_h, d_w, 1])
        biases = tf.get_variable('b', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.nn.bias_add(_deconv, biases)
        # deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
        return (deconv, w, biases) if with_w else deconv


def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    """Implements fully connected layer.
        Takes a (batch of) flattened image(s) as input, returns output_size values for each image"""
    shape = input_.get_shape().as_list()
    with tf.variable_scope(scope or "Linear"):
        # shape[1] is flattened image dims
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
                               initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias


def lrelu(x, leak=0.2):
    return tf.maximum(x, leak*x)


def concat(tensors, axis, *args, **kwargs):
    return tf.concat_v2(tensors, axis, *args, **kwargs) if "concat_v2" in dir(tf)\
        else tf.concat(tensors, axis, *args, **kwargs)


def conv_cond_concat(x, y):
    """Concatenate conditioning vector as additional feature maps."""
    x_shape = x.get_shape()
    y_shape = y.get_shape()
    return concat([x, y*tf.ones([x_shape[0], x_shape[1], x_shape[2], y_shape[3]])], 3)
