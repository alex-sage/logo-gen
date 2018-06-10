import tflib as lib
import tflib.ops.linear
import tflib.ops.cond_batchnorm
import tflib.ops.conv2d
import tflib.ops.deconv2d
import tflib.ops.batchnorm
import tflib.ops.layernorm
import tflib.ops.concat
from tflib.ops.gan_ops import *

import numpy as np
import tensorflow as tf


def Generator_Resnet_32(cfg, n_samples, labels, noise=None, is_training=True):
    if noise is None:
        noise = tf.random_normal([n_samples, 128])
    add_dim = 0
    if cfg.LAYER_COND:
        y = labels
        noise = tflib.ops.concat.concat([noise, y], 1)
        add_dim = cfg.N_LABELS
    output = lib.ops.linear.Linear('Generator.Input', 128 + add_dim, 4 * 4 * cfg.DIM_G, noise)
    output = tf.reshape(output, [-1, cfg.DIM_G, 4, 4])
    output = ResidualBlock(cfg, 'Generator.1', cfg.DIM_G, cfg.DIM_G, 3, output, resample='up', labels=labels,
                           is_training=is_training)
    output = ResidualBlock(cfg, 'Generator.2', cfg.DIM_G, cfg.DIM_G, 3, output, resample='up', labels=labels,
                           is_training=is_training)
    output = ResidualBlock(cfg, 'Generator.3', cfg.DIM_G, cfg.DIM_G, 3, output, resample='up', labels=labels,
                           is_training=is_training)
    output = Normalize(cfg, 'Generator.OutputN', output, is_training=is_training)
    output = nonlinearity(output)
    if cfg.LAYER_COND:
        yb = tf.reshape(y, [-1, cfg.N_LABELS, 1, 1])
        output = tflib.ops.concat.conv_cond_concat(output, yb)
    output = lib.ops.conv2d.Conv2D('Generator.Output', cfg.DIM_G + add_dim, 3, 3, output, he_init=False)
    output = tf.tanh(output)
    return tf.reshape(output, [-1, cfg.OUTPUT_DIM])


def Discriminator_Resnet_32(cfg, inputs, labels, noise=None, is_training=True):
    output = tf.reshape(inputs, [-1, 3, 32, 32])
    add_dim = 0
    if cfg.LAYER_COND:
        y = labels
        add_dim = cfg.N_LABELS
    output = OptimizedResBlockDisc1(cfg, output, labels=labels)
    output = ResidualBlock(cfg, 'Discriminator.2', cfg.DIM_D, cfg.DIM_D, 3, output, resample='down', labels=labels,
                           is_training=is_training)
    output = ResidualBlock(cfg, 'Discriminator.3', cfg.DIM_D, cfg.DIM_D, 3, output, resample=None, labels=labels,
                           is_training=is_training)
    output = ResidualBlock(cfg, 'Discriminator.4', cfg.DIM_D, cfg.DIM_D, 3, output, resample=None, labels=labels,
                           is_training=is_training)
    output = nonlinearity(output)
    output = tf.reduce_mean(output, axis=[2, 3])
    if cfg.LAYER_COND:
        output = tflib.ops.concat.concat([output, y], 1)
    output_wgan = lib.ops.linear.Linear('Discriminator.Output', cfg.DIM_D + add_dim, 1, output)
    output_wgan = tf.reshape(output_wgan, [-1])
    if cfg.CONDITIONAL and cfg.ACGAN:
        output_acgan = lib.ops.linear.Linear('Discriminator.ACGANOutput', cfg.DIM_D, cfg.N_LABELS, output)
        return output_wgan, output_acgan
    else:
        return output_wgan, None


def Generator_DCGAN_32(cfg, n_samples, labels, noise=None):
    if noise is None:
        noise = tf.random_normal([n_samples, 128])

    output = lib.ops.linear.Linear('Generator.Input', 128, 4 * 4 * 4 * cfg.DIM_G, noise)
    output = lib.ops.batchnorm.Batchnorm('Generator.BN1', [0], output)
    output = tf.nn.relu(output)
    output = tf.reshape(output, [-1, 4 * cfg.DIM_G, 4, 4])

    output = lib.ops.deconv2d.Deconv2D('Generator.2', 4 * cfg.DIM_G, 2 * cfg.DIM_G, 5, output)
    output = lib.ops.batchnorm.Batchnorm('Generator.BN2', [0, 2, 3], output)
    output = tf.nn.relu(output)

    output = lib.ops.deconv2d.Deconv2D('Generator.3', 2 * cfg.DIM_G, cfg.DIM_G, 5, output)
    output = lib.ops.batchnorm.Batchnorm('Generator.BN3', [0, 2, 3], output)
    output = tf.nn.relu(output)

    output = lib.ops.deconv2d.Deconv2D('Generator.5', cfg.DIM_G, 3, 5, output)

    output = tf.tanh(output)

    return tf.reshape(output, [-1, cfg.OUTPUT_DIM])


def Discriminator_DCGAN_32(cfg, inputs, labels):
    output = tf.reshape(inputs, [-1, 3, 32, 32])

    output = lib.ops.conv2d.Conv2D('Discriminator.1', 3, cfg.DIM_D, 5, output, stride=2)
    output = LeakyReLU(output)

    output = lib.ops.conv2d.Conv2D('Discriminator.2', cfg.DIM_D, 2 * cfg.DIM_D, 5, output, stride=2)
    if cfg.MODE != 'wgan-gp':
        output = lib.ops.batchnorm.Batchnorm('Discriminator.BN2', [0, 2, 3], output)
    output = LeakyReLU(output)

    output = lib.ops.conv2d.Conv2D('Discriminator.3', 2 * cfg.DIM_D, 4 * cfg.DIM_D, 5, output, stride=2)
    if cfg.MODE != 'wgan-gp':
        output = lib.ops.batchnorm.Batchnorm('Discriminator.BN3', [0, 2, 3], output)
    output = LeakyReLU(output)

    output = tf.reshape(output, [-1, 4 * 4 * 4 * cfg.DIM_D])
    output = lib.ops.linear.Linear('Discriminator.Output', 4 * 4 * 4 * cfg.DIM_D, 1, output)

    return tf.reshape(output, [-1]), None

# GoodGenerator
def Generator_Resnet_64(cfg, n_samples, labels, noise=None, is_training=True, dim=None, nonlinearity=tf.nn.relu):
    if dim is None:
        dim = cfg.DIM_G
    if noise is None:
        noise = tf.random_normal([n_samples, 128])
    if cfg.LAYER_COND:
        y = labels
        noise = tflib.ops.concat.concat([noise, y], 1)
        add_dim = cfg.N_LABELS
    else:
        add_dim = 0
    output = lib.ops.linear.Linear('Generator.Input', 128 + add_dim, 4*4*8*dim, noise)
    output = tf.reshape(output, [-1, 8*dim, 4, 4])

    output = ResidualBlock(cfg, 'Generator.Res1', 8*dim, 8*dim, 3, output, resample='up', labels=labels, is_training=is_training)
    output = ResidualBlock(cfg, 'Generator.Res2', 8*dim, 4*dim, 3, output, resample='up', labels=labels, is_training=is_training)
    output = ResidualBlock(cfg, 'Generator.Res3', 4*dim, 2*dim, 3, output, resample='up', labels=labels, is_training=is_training)
    output = ResidualBlock(cfg, 'Generator.Res4', 2*dim, 1*dim, 3, output, resample='up', labels=labels, is_training=is_training)

    output = Normalize(cfg, 'Generator.OutputN', output, is_training=is_training)
    output = tf.nn.relu(output)
    if cfg.LAYER_COND:
        yb = tf.reshape(y, [-1, cfg.N_LABELS, 1, 1])
        output = tflib.ops.concat.conv_cond_concat(output, yb)
    output = lib.ops.conv2d.Conv2D('Generator.Output', 1*dim + add_dim, 3, 3, output)
    output = tf.tanh(output)

    return tf.reshape(output, [-1, cfg.OUTPUT_DIM])

# GoodDiscriminator
def Discriminator_Resnet_64(cfg, inputs, labels, is_training=True,  dim=None):
    if dim is None:
        dim = cfg.DIM_D
    output = tf.reshape(inputs, [-1, 3, 64, 64])
    if cfg.LAYER_COND:
        y = labels
        add_dim = cfg.N_LABELS
        yb = tf.reshape(y, [-1, cfg.N_LABELS, 1, 1])
        output = tflib.ops.concat.conv_cond_concat(output, yb)
    else:
        add_dim = 0
    output = lib.ops.conv2d.Conv2D('Discriminator.Input', 3 + add_dim, dim, 3, output, he_init=False)

    output = ResidualBlock(cfg, 'Discriminator.Res1', dim, 2 * dim, 3, output, resample='down', labels=labels, is_training=is_training)
    output = ResidualBlock(cfg, 'Discriminator.Res2', 2 * dim, 4 * dim, 3, output, resample='down', labels=labels, is_training=is_training)
    output = ResidualBlock(cfg, 'Discriminator.Res3', 4 * dim, 8 * dim, 3, output, resample='down', labels=labels, is_training=is_training)
    output = ResidualBlock(cfg, 'Discriminator.Res4', 8 * dim, 8 * dim, 3, output, resample='down', labels=labels, is_training=is_training)

    output = tf.reshape(output, [-1, 4 * 4 * 8 * dim])
    if cfg.LAYER_COND:
        output = tflib.ops.concat.concat([output, y], 1)
    output_wgan = lib.ops.linear.Linear('Discriminator.Output', 4 * 4 * 8 * dim + add_dim, 1, output)

    output_wgan = tf.reshape(output_wgan, [-1])
    if cfg.CONDITIONAL and cfg.ACGAN:
        output_acgan = lib.ops.linear.Linear('Discriminator.ACGANOutput', 4 * 4 * 8 * dim, cfg.N_LABELS, output)
        return output_wgan, output_acgan
    else:
        return output_wgan, None




def Generator_DCGAN_64(cfg, n_samples, labels, noise=None, dim=None, bn=True, nonlinearity=tf.nn.relu):
    if dim is None:
        dim = cfg.DIM_G

    lib.ops.conv2d.set_weights_stdev(0.02)
    lib.ops.deconv2d.set_weights_stdev(0.02)
    lib.ops.linear.set_weights_stdev(0.02)

    if noise is None:
        noise = tf.random_normal([n_samples, 128])

    output = lib.ops.linear.Linear('Generator.Input', 128, 4*4*8*dim, noise)
    output = tf.reshape(output, [-1, 8*dim, 4, 4])
    if bn:
        output = Normalize(cfg, 'Generator.BN1', output)
    output = nonlinearity(output)

    output = lib.ops.deconv2d.Deconv2D('Generator.2', 8*dim, 4*dim, 5, output)
    if bn:
        output = Normalize(cfg, 'Generator.BN2', output)
    output = nonlinearity(output)

    output = lib.ops.deconv2d.Deconv2D('Generator.3', 4*dim, 2*dim, 5, output)
    if bn:
        output = Normalize(cfg, 'Generator.BN3', output)
    output = nonlinearity(output)

    output = lib.ops.deconv2d.Deconv2D('Generator.4', 2*dim, dim, 5, output)
    if bn:
        output = Normalize(cfg, 'Generator.BN4', output)
    output = nonlinearity(output)

    output = lib.ops.deconv2d.Deconv2D('Generator.5', dim, 3, 5, output)
    output = tf.tanh(output)

    lib.ops.conv2d.unset_weights_stdev()
    lib.ops.deconv2d.unset_weights_stdev()
    lib.ops.linear.unset_weights_stdev()

    return tf.reshape(output, [-1, cfg.OUTPUT_DIM])


def Discriminator_DCGAN_64(cfg, inputs, labels, dim=None, bn=True, nonlinearity=LeakyReLU):
    if dim is None:
        dim = cfg.DIM_D

    output = tf.reshape(inputs, [-1, 3, 64, 64])

    lib.ops.conv2d.set_weights_stdev(0.02)
    lib.ops.deconv2d.set_weights_stdev(0.02)
    lib.ops.linear.set_weights_stdev(0.02)

    output = lib.ops.conv2d.Conv2D('Discriminator.1', 3, dim, 5, output, stride=2)
    output = nonlinearity(output)

    output = lib.ops.conv2d.Conv2D('Discriminator.2', dim, 2*dim, 5, output, stride=2)
    if bn:
        output = Normalize(cfg, 'Discriminator.BN2', output)
    output = nonlinearity(output)

    output = lib.ops.conv2d.Conv2D('Discriminator.3', 2*dim, 4*dim, 5, output, stride=2)
    if bn:
        output = Normalize(cfg, 'Discriminator.BN3', output)
    output = nonlinearity(output)

    output = lib.ops.conv2d.Conv2D('Discriminator.4', 4*dim, 8*dim, 5, output, stride=2)
    if bn:
        output = Normalize(cfg, 'Discriminator.BN4', output)
    output = nonlinearity(output)

    output = tf.reshape(output, [-1, 4*4*8*dim])
    output = lib.ops.linear.Linear('Discriminator.Output', 4*4*8*dim, 1, output)

    lib.ops.conv2d.unset_weights_stdev()
    lib.ops.deconv2d.unset_weights_stdev()
    lib.ops.linear.unset_weights_stdev()

    return tf.reshape(output, [-1]), None


def Generator_MultiplicativeDCGAN_64(cfg, n_samples, labels, noise=None, dim=None, bn=True):
    if dim is None:
        dim = cfg.DIM_G
    if noise is None:
        noise = tf.random_normal([n_samples, 128])

    output = lib.ops.linear.Linear('Generator.Input', 128, 4*4*8*dim*2, noise)
    output = tf.reshape(output, [-1, 8*dim*2, 4, 4])
    if bn:
        output = Normalize(cfg, 'Generator.BN1', output)
    output = pixcnn_gated_nonlinearity(output[:,::2], output[:,1::2])

    output = lib.ops.deconv2d.Deconv2D('Generator.2', 8*dim, 4*dim*2, 5, output)
    if bn:
        output = Normalize(cfg, 'Generator.BN2', output)
    output = pixcnn_gated_nonlinearity(output[:,::2], output[:,1::2])

    output = lib.ops.deconv2d.Deconv2D('Generator.3', 4*dim, 2*dim*2, 5, output)
    if bn:
        output = Normalize(cfg, 'Generator.BN3', output)
    output = pixcnn_gated_nonlinearity(output[:,::2], output[:,1::2])

    output = lib.ops.deconv2d.Deconv2D('Generator.4', 2*dim, dim*2, 5, output)
    if bn:
        output = Normalize(cfg, 'Generator.BN4', output)
    output = pixcnn_gated_nonlinearity(output[:,::2], output[:,1::2])

    output = lib.ops.deconv2d.Deconv2D('Generator.5', dim, 3, 5, output)
    output = tf.tanh(output)

    return tf.reshape(output, [-1, cfg.OUTPUT_DIM])


def Discriminator_MultiplicativeDCGAN_64(cfg, inputs, labels, dim=None, bn=True):
    if dim is None:
        dim = cfg.DIM_D
    output = tf.reshape(inputs, [-1, 3, 64, 64])

    output = lib.ops.conv2d.Conv2D('Discriminator.1', 3, dim*2, 5, output, stride=2)
    output = pixcnn_gated_nonlinearity(output[:,::2], output[:,1::2])

    output = lib.ops.conv2d.Conv2D('Discriminator.2', dim, 2*dim*2, 5, output, stride=2)
    if bn:
        output = Normalize(cfg, 'Discriminator.BN2', output)
    output = pixcnn_gated_nonlinearity(output[:,::2], output[:,1::2])

    output = lib.ops.conv2d.Conv2D('Discriminator.3', 2*dim, 4*dim*2, 5, output, stride=2)
    if bn:
        output = Normalize(cfg, 'Discriminator.BN3', output)
    output = pixcnn_gated_nonlinearity(output[:,::2], output[:,1::2])

    output = lib.ops.conv2d.Conv2D('Discriminator.4', 4*dim, 8*dim*2, 5, output, stride=2)
    if bn:
        output = Normalize(cfg, 'Discriminator.BN4', output)
    output = pixcnn_gated_nonlinearity(output[:,::2], output[:,1::2])

    output = tf.reshape(output, [-1, 4*4*8*dim])
    output = lib.ops.linear.Linear('Discriminator.Output', 4*4*8*dim, 1, output)

    return tf.reshape(output, [-1]), None


def Generator_Bottleneck_Resnet_64(cfg, n_samples, labels, noise=None, dim=None):
    if dim is None:
        dim = cfg.DIM_G
    if noise is None:
        noise = tf.random_normal([n_samples, 128])

    output = lib.ops.linear.Linear('Generator.Input', 128, 4*4*8*dim, noise)
    output = tf.reshape(output, [-1, 8*dim, 4, 4])

    for i in xrange(6):
        output = BottleneckResidualBlock('Generator.4x4_{}'.format(i), 8*dim, 8*dim, 3, output, resample=None)
    output = BottleneckResidualBlock('Generator.Up1', 8*dim, 4*dim, 3, output, resample='up')
    for i in xrange(6):
        output = BottleneckResidualBlock('Generator.8x8_{}'.format(i), 4*dim, 4*dim, 3, output, resample=None)
    output = BottleneckResidualBlock('Generator.Up2', 4*dim, 2*dim, 3, output, resample='up')
    for i in xrange(6):
        output = BottleneckResidualBlock('Generator.16x16_{}'.format(i), 2*dim, 2*dim, 3, output, resample=None)
    output = BottleneckResidualBlock('Generator.Up3', 2*dim, 1*dim, 3, output, resample='up')
    for i in xrange(6):
        output = BottleneckResidualBlock('Generator.32x32_{}'.format(i), 1*dim, 1*dim, 3, output, resample=None)
    output = BottleneckResidualBlock('Generator.Up4', 1*dim, dim/2, 3, output, resample='up')
    for i in xrange(5):
        output = BottleneckResidualBlock('Generator.64x64_{}'.format(i), dim/2, dim/2, 3, output, resample=None)

    output = lib.ops.conv2d.Conv2D('Generator.Out', dim/2, 3, 1, output, he_init=False)
    output = tf.tanh(output / 5.)

    return tf.reshape(output, [-1, cfg.OUTPUT_DIM])


def Discriminator_Bottleneck_Resnet_64(cfg, inputs, labels, dim=None):
    if dim is None:
        dim = cfg.DIM_D
    output = tf.reshape(inputs, [-1, 3, 64, 64])
    output = lib.ops.conv2d.Conv2D('Discriminator.In', 3, dim/2, 1, output, he_init=False)

    for i in xrange(5):
        output = BottleneckResidualBlock('Discriminator.64x64_{}'.format(i), dim/2, dim/2, 3, output, resample=None)
    output = BottleneckResidualBlock('Discriminator.Down1', dim/2, dim*1, 3, output, resample='down')
    for i in xrange(6):
        output = BottleneckResidualBlock('Discriminator.32x32_{}'.format(i), dim*1, dim*1, 3, output, resample=None)
    output = BottleneckResidualBlock('Discriminator.Down2', dim*1, dim*2, 3, output, resample='down')
    for i in xrange(6):
        output = BottleneckResidualBlock('Discriminator.16x16_{}'.format(i), dim*2, dim*2, 3, output, resample=None)
    output = BottleneckResidualBlock('Discriminator.Down3', dim*2, dim*4, 3, output, resample='down')
    for i in xrange(6):
        output = BottleneckResidualBlock('Discriminator.8x8_{}'.format(i), dim*4, dim*4, 3, output, resample=None)
    output = BottleneckResidualBlock('Discriminator.Down4', dim*4, dim*8, 3, output, resample='down')
    for i in xrange(6):
        output = BottleneckResidualBlock('Discriminator.4x4_{}'.format(i), dim*8, dim*8, 3, output, resample=None)

    output = tf.reshape(output, [-1, 4*4*8*dim])
    output = lib.ops.linear.Linear('Discriminator.Output', 4*4*8*dim, 1, output)

    return tf.reshape(output / 5., [-1]), None


def Generator_FC_64(cfg, n_samples, labels, noise=None, FC_DIM=512):
    if noise is None:
        noise = tf.random_normal([n_samples, 128])

    output = ReLULayer('Generator.1', 128, FC_DIM, noise)
    output = ReLULayer('Generator.2', FC_DIM, FC_DIM, output)
    output = ReLULayer('Generator.3', FC_DIM, FC_DIM, output)
    output = ReLULayer('Generator.4', FC_DIM, FC_DIM, output)
    output = lib.ops.linear.Linear('Generator.Out', FC_DIM, cfg.OUTPUT_DIM, output)

    output = tf.tanh(output)

    return output


def Discriminator_FC_64(cfg, inputs, labels, FC_DIM=512, n_layers=3):
    output = LeakyReLULayer('Discriminator.Input', cfg.OUTPUT_DIM, FC_DIM, inputs)
    for i in xrange(n_layers):
        output = LeakyReLULayer('Discriminator.{}'.format(i), FC_DIM, FC_DIM, output)
    output = lib.ops.linear.Linear('Discriminator.Out', FC_DIM, 1, output)

    return tf.reshape(output, [-1]), None


DIM_G_64  = 64
DIM_G_32  = 128
DIM_G_16  = 256
DIM_G_8   = 512
DIM_G_4   = 512

# DIM_D_64  = 128
# DIM_D_32  = 256
# DIM_D_16  = 512
# DIM_D_8   = 1024
# DIM_D_4   = 1024
DIM_D_64  = 64
DIM_D_32  = 128
DIM_D_16  = 256
DIM_D_8   = 512
DIM_D_4   = 512

def Generator_Resnet_128(cfg, n_samples, labels, noise=None, is_training=True):
    if noise is None:
        noise = tf.random_normal([n_samples, 128])
    add_dim = 0
    if cfg.LAYER_COND:
        y = labels
        noise = tflib.ops.concat.concat([noise, y], 1)
        add_dim = cfg.N_LABELS
    output = lib.ops.linear.Linear('Generator.Input', 128 + add_dim, 4*4*DIM_G_4, noise)
    output = tf.reshape(output, [-1, DIM_G_4, 4, 4])

    # output = ResidualBlock('Generator.4_1', DIM_G_4, DIM_G_4, 3, output, resample=None)
    # output = ResidualBlock('Generator.4_2', DIM_G_4, DIM_G_4, 3, output, resample=None)
    output = ResidualBlock(cfg, 'Generator.4_3', DIM_G_4, DIM_G_8, 3, output, resample='up', labels=labels,
                           is_training=is_training)

    # output = ResidualBlock('Generator.8_1', DIM_G_8, DIM_G_8, 3, output, resample=None)
    # output = ResidualBlock('Generator.8_2', DIM_G_8, DIM_G_8, 3, output, resample=None)
    output = ResidualBlock(cfg, 'Generator.8_3', DIM_G_8, DIM_G_16, 3, output, resample='up', labels=labels,
                           is_training=is_training)

    # output = ResidualBlock('Generator.16_1', DIM_G_16, DIM_G_16, 3, output, resample=None)
    # output = ResidualBlock('Generator.16_2', DIM_G_16, DIM_G_16, 3, output, resample=None)
    output = ResidualBlock(cfg, 'Generator.16_3', DIM_G_16, DIM_G_32, 3, output, resample='up', labels=labels,
                           is_training=is_training)

    # output = ResidualBlock('Generator.32_1', DIM_G_32, DIM_G_32, 3, output, resample=None)
    # output = ResidualBlock('Generator.32_2', DIM_G_32, DIM_G_32, 3, output, resample=None)
    output = ResidualBlock(cfg, 'Generator.32_3', DIM_G_32, DIM_G_64, 3, output, resample='up', labels=labels,
                           is_training=is_training)

    output = Normalize(cfg, 'Generator.OutputN', output,  is_training=is_training)
    output = nonlinearity(output)
    if cfg.LAYER_COND:
        yb = tf.reshape(y, [-1, cfg.N_LABELS, 1, 1])
        output = tflib.ops.concat.conv_cond_concat(output, yb)
    output = ScaledUpsampleConv('Generator.Output', DIM_G_64 + add_dim, 3, 5, output, he_init=False)
    # output = lib.ops.deconv2d.Deconv2D('Generator.Output', DIM_G_64, 3, 5, output, he_init=False)

    output = tf.tanh(output)

    return tf.reshape(output, [-1, cfg.OUTPUT_DIM])

def Discriminator_Resnet_128(cfg, inputs, labels, is_training=True):
    add_dim = 0
    output = tf.reshape(inputs, [-1, 3, 128, 128])
    if cfg.LAYER_COND:
        y = labels
        add_dim = cfg.N_LABELS
        yb = tf.reshape(y, [-1, cfg.N_LABELS, 1, 1])
        output = tflib.ops.concat.conv_cond_concat(output, yb)
    output = lib.ops.conv2d.Conv2D('Discriminator.Input', 3 + add_dim, DIM_D_64, 5, output, he_init=True, stride=2)

    # output = ResidualBlock('Discriminator.64_1', DIM_D_64, DIM_D_64, 3, output, resample=None)
    # output = ResidualBlock('Discriminator.64_2', DIM_D_64, DIM_D_64, 3, output, resample=None)
    output = ResidualBlock(cfg, 'Discriminator.64_3', DIM_D_64, DIM_D_32, 3, output, resample='down', labels=labels,
                           is_training=is_training)

    # output = ResidualBlock('Discriminator.32_1', DIM_D_32, DIM_D_32, 3, output, resample=None)
    # output = ResidualBlock('Discriminator.32_2', DIM_D_32, DIM_D_32, 3, output, resample=None)
    output = ResidualBlock(cfg, 'Discriminator.32_3', DIM_D_32, DIM_D_16, 3, output, resample='down', labels=labels,
                           is_training=is_training)

    # output = ResidualBlock('Discriminator.16_1', DIM_D_16, DIM_D_16, 3, output, resample=None)
    # output = ResidualBlock('Discriminator.16_2', DIM_D_16, DIM_D_16, 3, output, resample=None)
    output = ResidualBlock(cfg, 'Discriminator.16_3', DIM_D_16, DIM_D_8, 3, output, resample='down', labels=labels,
                           is_training=is_training)

    output = ResidualBlock(cfg, 'Discriminator.8_1', DIM_D_8, DIM_D_8, 3, output, resample=None, labels=labels,
                           is_training=is_training)
    output = ResidualBlock(cfg, 'Discriminator.8_2', DIM_D_8, DIM_D_8, 3, output, resample=None, labels=labels,
                           is_training=is_training)
    # output = ResidualBlock('Discriminator.8_3', DIM_D_8, DIM_D_4, 3, output, resample='down')

    # output = ResidualBlock('Discriminator.4_1', DIM_D_4, DIM_D_4, 3, output, resample=None)
    # output = ResidualBlock('Discriminator.4_2', DIM_D_4, DIM_D_4, 3, output, resample=None)

    # output = Normalize('Discriminator.OutputN', output)
    # output = output / 10.
    output = tf.reduce_mean(output, axis=[2,3])
    if cfg.LAYER_COND:
        output = tflib.ops.concat.concat([output, y], 1)
    output_wgan = lib.ops.linear.Linear('Discriminator.Output', DIM_D_8 + add_dim, 1, output)

    # output = Normalize('Discriminator.OutputN', output)
    # output = nonlinearity(output)
    # output = tf.reshape(output, [-1, 4*4*DIM_D_4])
    # output = lib.ops.linear.Linear('Discriminator.Output', 4*4*DIM_D_4, 1, output)

    output_wgan = tf.reshape(output_wgan, [-1])
    if cfg.CONDITIONAL and cfg.ACGAN:
        output_acgan = lib.ops.linear.Linear('Discriminator.ACGANOutput', DIM_D_8, cfg.N_LABELS, output)
        return output_wgan, output_acgan
    else:
        return output_wgan, None