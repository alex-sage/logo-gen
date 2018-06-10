import tensorflow as tf
import numpy as np

import tflib as lib
import tflib.ops.linear
import tflib.ops.cond_batchnorm
import tflib.ops.conv2d
import tflib.ops.deconv2d
import tflib.ops.batchnorm
import tflib.ops.layernorm
import tflib.ops.concat

import functools


def nonlinearity(x):
    return tf.nn.relu(x)


def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha*x, x)


def ReLULayer(name, n_in, n_out, inputs):
    # for 32p initialization was not set!
    output = lib.ops.linear.Linear(name+'.Linear', n_in, n_out, inputs, initialization='he')
    return tf.nn.relu(output)


def LeakyReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(name+'.Linear', n_in, n_out, inputs, initialization='he')
    return LeakyReLU(output)


def Normalize(cfg, name, inputs, labels=None, is_training=True):
    """This is messy, but basically it chooses between batchnorm, layernorm, 
    their conditional variants, or nothing, depending on the value of `name` and
    the global hyperparam flags."""
    if (not cfg.CONDITIONAL) or cfg.LAYER_COND:
        labels = None
    if cfg.CONDITIONAL and cfg.ACGAN and ('Discriminator' in name):
        labels = None

    if ('Discriminator' in name) and cfg.NORMALIZATION_D:
        if labels is not None:
            # todo: fix (does not work)
            # return lib.ops.layernorm.Layernorm_cond(name,[1,2,3],inputs,labels=labels,n_labels=N_LABELS)
            return lib.ops.cond_batchnorm.Batchnorm(name, [0, 2, 3], inputs, labels=labels, n_labels=cfg.N_LABELS)
        elif cfg.MODE == 'wgan-gp':
            return lib.ops.layernorm.Layernorm(name,[1,2,3],inputs)
        else:
            return tf.layers.batch_normalization(inputs, axis=1, training=is_training, fused=True)

    elif ('Generator' in name) and cfg.NORMALIZATION_G:
        if labels is not None:
            return lib.ops.cond_batchnorm.Batchnorm(name, [0,2,3], inputs,labels=labels, n_labels=cfg.N_LABELS)
        else:
            # return lib.ops.batchnorm.Batchnorm(name,[0,2,3], inputs, fused=True,
            #                                    is_training=is_training, stats_iter=stats_iter,
            #                                    update_moving_stats=update_moving_stats)
            return tf.layers.batch_normalization(inputs, axis=1, training=is_training, fused=True)
    else:
        return inputs


def pixcnn_gated_nonlinearity(a, b):
    return tf.sigmoid(a) * tf.tanh(b)


def SubpixelConv2D(*args, **kwargs):
    kwargs['output_dim'] = 4*kwargs['output_dim']
    output = lib.ops.conv2d.Conv2D(*args, **kwargs)
    output = tf.transpose(output, [0,2,3,1])
    output = tf.depth_to_space(output, 2)
    output = tf.transpose(output, [0,3,1,2])
    return output


def ConvMeanPool(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = lib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, inputs, he_init=he_init, biases=biases)
    output = tf.add_n([output[:,:,::2,::2], output[:,:,1::2,::2], output[:,:,::2,1::2], output[:,:,1::2,1::2]]) / 4.
    return output


def MeanPoolConv(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = inputs
    output = tf.add_n([output[:,:,::2,::2], output[:,:,1::2,::2], output[:,:,::2,1::2], output[:,:,1::2,1::2]]) / 4.
    output = lib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, output, he_init=he_init, biases=biases)
    return output


def UpsampleConv(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = inputs
    output = tf.concat([output, output, output, output], axis=1)
    output = tf.transpose(output, [0,2,3,1])
    output = tf.depth_to_space(output, 2)
    output = tf.transpose(output, [0,3,1,2])
    output = lib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, output, he_init=he_init, biases=biases)
    return output


def ResidualBlock(cfg, name, input_dim, output_dim, filter_size, inputs, resample=None, no_dropout=False, labels=None,
                  is_training=True):
    """
    resample: None, 'down', or 'up'
    """
    if cfg.LAYER_COND:
        y = labels
        add_dim = cfg.N_LABELS
        yb = tf.reshape(y, [-1, cfg.N_LABELS, 1, 1])
    else:
        add_dim = 0

    if resample=='down':
        conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim + add_dim, output_dim=input_dim)
        conv_2        = functools.partial(ConvMeanPool, input_dim=input_dim + add_dim, output_dim=output_dim)
        conv_shortcut = ConvMeanPool
    elif resample=='up':
        conv_1        = functools.partial(UpsampleConv, input_dim=input_dim + add_dim, output_dim=output_dim)
        conv_shortcut = UpsampleConv
        conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=output_dim + add_dim, output_dim=output_dim)
    elif resample==None:
        conv_shortcut = lib.ops.conv2d.Conv2D
        conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim + add_dim, output_dim=output_dim)
        conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=output_dim + add_dim, output_dim=output_dim)
    else:
        raise Exception('invalid resample value')

    if output_dim==input_dim and resample==None:
        shortcut = inputs # Identity skip-connection
    else:
        shortcut = conv_shortcut(name+'.Shortcut', input_dim=input_dim, output_dim=output_dim, filter_size=1, he_init=False, biases=True, inputs=inputs)

    output = inputs
    output = Normalize(cfg, name+'.N1', output, labels=labels, is_training=is_training)
    output = nonlinearity(output)
    if cfg.LAYER_COND:
        output = tflib.ops.concat.conv_cond_concat(output, yb)
    output = conv_1(name+'.Conv1', filter_size=filter_size, inputs=output)
    output = Normalize(cfg, name+'.N2', output, labels=labels, is_training=is_training)
    output = nonlinearity(output)
    if cfg.LAYER_COND:
        output = tflib.ops.concat.conv_cond_concat(output, yb)
    output = conv_2(name+'.Conv2', filter_size=filter_size, inputs=output)

    return shortcut + output


def OptimizedResBlockDisc1(cfg, inputs, labels):
    if cfg.LAYER_COND:
        y = labels
        add_dim = cfg.N_LABELS
        yb = tf.reshape(y, [-1, cfg.N_LABELS, 1, 1])
    else:
        add_dim = 0
    conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=3 + add_dim, output_dim=cfg.DIM_D)
    conv_2        = functools.partial(ConvMeanPool, input_dim=cfg.DIM_D + add_dim, output_dim=cfg.DIM_D)
    conv_shortcut = MeanPoolConv
    shortcut = conv_shortcut('Discriminator.1.Shortcut', input_dim=3, output_dim=cfg.DIM_D, filter_size=1,
                             he_init=False, biases=True, inputs=inputs)

    output = inputs
    if cfg.LAYER_COND:
        output = tflib.ops.concat.conv_cond_concat(output, yb)
    output = conv_1('Discriminator.1.Conv1', filter_size=3, inputs=output)
    output = nonlinearity(output)
    if cfg.LAYER_COND:
        output = tflib.ops.concat.conv_cond_concat(output, yb)
    output = conv_2('Discriminator.1.Conv2', filter_size=3, inputs=output)
    return shortcut + output


def BottleneckResidualBlock(name, input_dim, output_dim, filter_size, inputs, resample=None, he_init=True):
    """
    resample: None, 'down', or 'up'
    """
    if resample=='down':
        conv_shortcut = functools.partial(lib.ops.conv2d.Conv2D, stride=2)
        conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=input_dim/2)
        conv_1b       = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim/2, output_dim=output_dim/2, stride=2)
        conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=output_dim/2, output_dim=output_dim)
    elif resample=='up':
        conv_shortcut = SubpixelConv2D
        conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=input_dim/2)
        conv_1b       = functools.partial(lib.ops.deconv2d.Deconv2D, input_dim=input_dim/2, output_dim=output_dim/2)
        conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=output_dim/2, output_dim=output_dim)
    elif resample==None:
        conv_shortcut = lib.ops.conv2d.Conv2D
        conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim,  output_dim=input_dim/2)
        conv_1b       = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim/2,  output_dim=output_dim/2)
        conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim/2, output_dim=output_dim)

    else:
        raise Exception('invalid resample value')

    if output_dim==input_dim and resample is None:
        shortcut = inputs # Identity skip-connection
    else:
        shortcut = conv_shortcut(name+'.Shortcut', input_dim=input_dim, output_dim=output_dim, filter_size=1,
                                 he_init=False, biases=True, inputs=inputs)

    output = inputs
    output = tf.nn.relu(output)
    output = conv_1(name+'.Conv1', filter_size=1, inputs=output, he_init=he_init)
    output = tf.nn.relu(output)
    output = conv_1b(name+'.Conv1B', filter_size=filter_size, inputs=output, he_init=he_init)
    output = tf.nn.relu(output)
    output = conv_2(name+'.Conv2', filter_size=1, inputs=output, he_init=he_init, biases=False)
    output = Normalize(name+'.BN', [0,2,3], output)

    return shortcut + (0.3*output)


def ScaledUpsampleConv(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = inputs
    output = lib.ops.concat.concat([output, output, output, output], axis=1)
    output = tf.transpose(output, [0,2,3,1])
    output = tf.depth_to_space(output, 2)
    output = tf.transpose(output, [0,3,1,2])
    output = lib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, output, he_init=he_init, biases=biases, gain=0.5)
    return output
