import tensorflow as tf
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from ops import concat


def gauss_kernel_fixed(sigma, N):
    # Non-Adaptive kernel size
    if sigma == 0:
        return np.eye(2 * N + 1)[N]
    x = np.arange(-N, N + 1, 1.0)
    g = np.exp(-x * x / (2 * sigma * sigma))
    g = g / np.sum(np.abs(g))
    return g


def gaussian_blur(image, kernel, kernel_size, cdim=3):
    # kernel as placeholder variable, so it can change
    outputs = []
    pad_w = (kernel_size - 1) // 2
    padded = tf.pad(image, [[0, 0], [pad_w, pad_w], [pad_w, pad_w], [0, 0]], mode='REFLECT')
    for channel_idx in range(cdim):
        data_c = padded[:, :, :, channel_idx:(channel_idx + 1)]
        g = tf.reshape(kernel, [1, kernel_size, 1, 1])
        data_c = tf.nn.conv2d(data_c, g, [1, 1, 1, 1], 'VALID')
        g = tf.reshape(kernel, [kernel_size, 1, 1, 1])
        data_c = tf.nn.conv2d(data_c, g, [1, 1, 1, 1], 'VALID')
        outputs.append(data_c)
    return concat(outputs, axis=3)


def gauss_kernel(sigma, eps, truncate):
    # Adaptive kernel size based on sigma,
    # for fixed kernel size, hardcode N
    # truncate limits kernel size as in scipy's gaussian_filter

    N = np.clip(np.ceil(sigma * np.sqrt(2 * np.log(1 / eps))), 1, truncate)
    x = np.arange(-N, N + 1, 1.0)
    g = np.exp(-x * x / (2 * sigma * sigma))
    g = g / np.sum(np.abs(g))
    return g


def gaussian_blur_adaptive(image, sigma, eps=0.01, img_width=32, cdim=3):
    if sigma == 0:
        return image
    outputs = []
    kernel = gauss_kernel(sigma, eps, img_width - 1)
    pad_w = (kernel.shape[0] - 1) // 2
    padded = tf.pad(image, [[0, 0], [pad_w, pad_w], [pad_w, pad_w], [0, 0]], mode='REFLECT')
    for channel_idx in range(cdim):
        data_c = padded[:, :, :, channel_idx:(channel_idx + 1)]
        g = np.expand_dims(kernel, 0)
        g = np.expand_dims(g, axis=2)
        g = np.expand_dims(g, axis=3)
        data_c = tf.nn.conv2d(data_c, g, [1, 1, 1, 1], 'VALID')
        g = np.expand_dims(kernel, 1)
        g = np.expand_dims(g, axis=2)
        g = np.expand_dims(g, axis=3)
        data_c = tf.nn.conv2d(data_c, g, [1, 1, 1, 1], 'VALID')
        outputs.append(data_c)
    return concat(outputs, axis=3)


# --- old functions, not used any more --

def _gkern(sigma, truncate=2, dim=1):
    """Returns a 1D or 2D Gaussian kernel array."""
    size = truncate * 2 + 1
    if dim == 1:
        delta = np.eye(size)[truncate]
    if dim == 2:
        # create nxn zeros
        delta = np.zeros((size, size))
        # set element at the middle to one, a kronecker delta
        delta[truncate, truncate] = 1
    # gaussian-smooth the dirac, resulting in a gaussian filter mask
    return gaussian_filter(delta, sigma, truncate=truncate, mode='constant').astype('float32')


def _gauss_blur_tensor(img_batch, kernel, img_batch_dim, k_size):
    blur_gauss_kernel = tf.stack([kernel, kernel, kernel])
    blur_gauss_kernel_4d = tf.reshape(blur_gauss_kernel, (k_size, k_size, img_batch_dim[3], 1))
    output = tf.nn.depthwise_conv2d(img_batch,
                                      blur_gauss_kernel_4d,
                                      strides=[1, 1, 1, 1],
                                      padding='SAME')
    output = tf.reshape(output, [img_batch_dim[0], img_batch_dim[1], img_batch_dim[2], img_batch_dim[3]])
    # print('blur %f' % kernel[1, 1])
    return output


def _gauss_blur(img_batch, sigma, truncate=2):
    return np.array([gaussian_filter(image, (sigma, sigma, 0), truncate=truncate, mode='constant').astype('float32')
                     for image in img_batch])
