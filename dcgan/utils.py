"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import os
import math
import pprint
import scipy.misc
import scipy.stats as stats

import numpy as np
from time import gmtime, strftime
from six.moves import xrange
# from matplotlib import pyplot as plt

import tensorflow as tf
import tensorflow.contrib.slim as slim

import sys
sys.path.insert(0, '../image-tools')
import metrics
from gauss import gauss_kernel_fixed

pp = pprint.PrettyPrinter()


def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)


def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))


def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)


def imsave(images, size, path):
    if images.shape[3] == 1:
        merged = merge(images, size)
        return scipy.misc.imsave(path, merged.reshape(merged.shape[:2]))
    return scipy.misc.imsave(path, merge(images, size))


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    c = images.shape[3]
    img = np.zeros((h * size[0], w * size[1], c))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image
    return img


def inverse_transform(images):
    return (images+1.)/2.


def slerp(val, low, high):
    """Spherical interpolation. val has a range of 0 to 1.
        Taken from https://github.com/dribnet/plat"""
    if val <= 0:
        return low
    elif val >= 1:
        return high
    elif np.allclose(low, high):
        return low
    omega = np.arccos(np.dot(low/np.linalg.norm(low), high/np.linalg.norm(high)))
    so = np.sin(omega)
    return np.sin((1.0-val)*omega) / so * low + np.sin(val*omega)/so * high


def interpolate(sess, dcgan, z_start, z_stop, n_steps=62, y_start=None, y_stop=None, transform=True):
    """Interpolates between two samples in z-space
    
    Input parameters:
    sess: TF session
    dcgan: DCGAN object for sampling
    z_start: z-vector of the first sample
    z_start: z-vector of the second sample
    n_steps: number of intermediate samples to produce
    sampling: the sampling method used for training ['uniform']
    transform: if True, the pixel values will be transformed to their normal image range [True]
    y_start: label for first sample (numerical)
    y_stop: label for second sample (numerical)
    
    RETURNS an array of n_steps+2 samples"""
    y_dim = 0
    if y_start is not None:
        y_dim = dcgan.y_dim
        if y_stop is None:
            y_stop = y_start
        if y_start != y_stop:
            z_start = np.concatenate((z_start, np.eye(y_dim)[y_start]))
            z_stop = np.concatenate((z_stop, np.eye(y_dim)[y_stop]))
    # limit to batch size for simplicity
    if n_steps > (dcgan.batch_size - 2):
        n_steps = dcgan.batch_size - 2
    # sample along big circle for all distributions
    steps = np.linspace(0, 1, n_steps + 2)
    z_samples = [slerp(step, z_start, z_stop) for step in steps]
    gauss_filter = gauss_kernel_fixed(dcgan.gauss_sigma, (dcgan.kernel_size - 1) // 2)
    if n_steps != (dcgan.batch_size - 2):
        z_samples += [np.zeros(dcgan.z_dim + y_dim) for i in range(dcgan.batch_size - n_steps - 2)]
    if y_dim > 0:
        if y_start != y_stop:
            samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: np.array(z_samples)[:, :dcgan.z_dim],
                                                         dcgan.y: np.array(z_samples)[:, dcgan.z_dim:],
                                                         dcgan.gauss_kernel: gauss_filter},)[:n_steps + 2]
        else:
            samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: np.array(z_samples),
                                                         dcgan.y: np.eye(y_dim)
                                                         [np.full(dcgan.batch_size, y_start)],
                                                         dcgan.gauss_kernel: gauss_filter})[:n_steps + 2]
    else:
         samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: np.array(z_samples),
                                                      dcgan.gauss_kernel: gauss_filter})[:n_steps+2]
    if transform:
        samples = np.array([((sample + 1) / 2 * 255).astype(np.uint8) for sample in samples])
    return samples


def interactive_interp(sess, dcgan, config, sampling='uniform'):
    while True:
        z_samples = dcgan.z_sampler(config)
        has_labels = False
        try:
            if dcgan.has_labels:
                has_labels = True
                label = int(raw_input('Class label for first sample: '))
                sample_labels = np.eye(dcgan.y_dim)[np.full(dcgan.batch_size, label)]
        except Exception: pass
        gauss_filter = gauss_kernel_fixed(config.gauss_sigma, config.gauss_trunc)
        if has_labels:
            samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_samples, dcgan.y: sample_labels,
                                                         dcgan.gauss_kernel: gauss_filter})
        else:
            samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_samples, dcgan.gauss_kernel: gauss_filter})
        samples = np.array([((sample + 1) / 2 * 255).astype(np.uint8) for sample in samples])
        grid_size = int(math.ceil(math.sqrt(dcgan.batch_size)))
        scipy.misc.imshow(merge(samples, (grid_size, grid_size)))
        # from IPython import embed; embed()
        start = int(raw_input('First sample number: '))
        if has_labels:
            label2 = raw_input('Class label for second sample [same]: ')
            if label2 == '':
                label2 = label
                same = True
            else:
                label2 = int(label2)
                same = False
            sample_labels2 = np.eye(dcgan.y_dim)[np.full(dcgan.batch_size, label2)]
            if same:
                samples2 = samples
            else:
                samples2 = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_samples, dcgan.y: sample_labels2,
                                    dcgan.gauss_kernel: gauss_filter})
            scipy.misc.imshow(merge(samples2, (grid_size, grid_size)))
        stop = int(raw_input('Second sample number: '))
        n_steps = raw_input('Number of steps [62]: ')
        if n_steps == '':
            n_steps = 62
        else:
            n_steps = int(n_steps)
        if has_labels:
            series = interpolate(sess, dcgan, z_start=z_samples[start - 1], z_stop=z_samples[stop - 1],
                                 n_steps=n_steps, y_start=label, y_stop=label2, transform=True)
        else:
            series = interpolate(sess, dcgan, z_start=z_samples[start-1], z_stop=z_samples[stop-1],
                                 n_steps=n_steps, transform=True)
        scipy.misc.imshow(merge(series, (int(math.ceil((n_steps + 2) / 8)), 8)))
        c = raw_input('Continue? [y/n]')
        if c != 'y':
            break

def visualize(sess, dcgan, config, option):
    image_frame_dim = int(math.ceil(config.batch_size ** .5))
    # produce sample uniformly with nearest neighbour
    # option 0: additionally sort according to distance
    if (option == 1) or (option == 0):
        n_images = 20
        has_labels = False
        try:
            if dcgan.has_labels:
                # generate one image for each cluster / category
                has_labels = True
                if option == 0:
                    n_images = dcgan.y_dim
        except Exception: pass
        # sample DCGAN from uniform distribution in z
        print('sampling...')
        z_samples = dcgan.z_sampler(config)
        if has_labels:
            y_samples = np.eye(dcgan.y_dim)[np.random.choice(dcgan.y_dim, [n_images, config.batch_size])]
            samples = (z_samples, y_samples)
            samples = np.array([sess.run(dcgan.sampler, {dcgan.z: batch, dcgan.y: batch_y})
                                for batch, batch_y in samples])
        else:
            samples = np.array([sess.run(dcgan.sampler, feed_dict={dcgan.z: batch}) for batch in z_samples])
        # transform back to normal image value range and reshape to one array instead of batches
        print('transforming...')
        samples = np.array([((sample + 1) / 2 * 255).astype(np.uint8) for sample in samples]) \
            .reshape((samples.shape[0] * samples.shape[1],) + samples.shape[2:])
        # load and rescale training data to same size as samples
        print('loading and transforming orig data...')
        orig_data, _ = fh.load_icon_data(config.data_dir)
        orig_data = np.array(
            [scipy.misc.imresize(icon, (config.output_height, config.output_height)) for icon in orig_data])
        # get nearest neighbour indices from training set
        if option == 1:
            print('getting nearest neighbours...')
            nearest_idxs = metrics.nearest_icons(samples, orig_data)
        else:
            print('getting nearest neighbours...')
            nearest_idxs, distances = metrics.nearest_icons(samples, orig_data, get_dist=True)
            print('sorting...')
            # normalize distance over whole image content to prevent predominantly white images having low distance
            norms = np.sqrt(np.sum(np.power(samples, 2), axis=(1, 2, 3)))
            distances = np.array([distance / n for distance, n in zip(distances, norms)])
            sorting = np.argsort(distances)
            # import ipdb; ipdb.set_trace()
            samples = samples[sorting]
            nearest_idxs = np.array(nearest_idxs)[sorting]
        bs = config.batch_size
        for idx in xrange(n_images):
            print(" [*] %d" % idx)
            combined = []
            # combine samples and nearest neighbours for each batch and save as png
            for sample, orig in zip(samples[idx * bs:(idx + 1) * bs], orig_data[nearest_idxs[idx * bs:(idx + 1) * bs]]):
                combined += [sample, orig]
            scipy.misc.imsave(os.path.join(config.sample_dir, 'test_uniform_nearest_%s.png' % (idx)),
                              merge(np.array(combined), [image_frame_dim, image_frame_dim * 2]))
    # sample with uniform distribution
    if option == 2:
        n_images = 20
        has_labels = False
        try:
            if dcgan.has_labels:
                # generate one image for each cluster / category
                n_images = dcgan.y_dim
                has_labels = True
        except Exception: pass
        for idx in xrange(n_images):
            print(" [*] %d" % idx)
            z_sample = dcgan.z_sampler(config)
            # create gaussian convolution kernel as defined in run parameters
            kernel = gauss_kernel_fixed(config.gauss_sigma, config.gauss_trunc)
            if has_labels:
                # y = np.random.choice(dcgan.y_dim, config.batch_size)
                # y_one_hot = np.zeros((config.batch_size, dcgan.y_dim))
                # y_one_hot[np.arange(config.batch_size), y] = 1
                y_one_hot = np.eye(dcgan.y_dim)[np.full(config.batch_size, idx)]
                # print(y_one_hot)
                samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample, dcgan.y: y_one_hot, dcgan.gauss_kernel: kernel})
            else:
                samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample, dcgan.gauss_kernel: kernel})

            save_images(samples, [image_frame_dim, image_frame_dim],
                        os.path.join(config.sample_dir, 'test_uniform_%s.png' % (idx)))
    # sample with normal distribution
    if option == 3:
        for idx in xrange(100):
            print(" [*] %d" % idx)
            z_sample = np.random.normal(size=(config.batch_size, dcgan.z_dim))
            samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
            save_images(samples, [image_frame_dim, image_frame_dim], './samples/test_normal_%s.png' % (idx))
    # single sample with uniform distribution
    if option == 4:
        z_sample = np.random.uniform(-0.5, 0.5, size=(config.batch_size, dcgan.z_dim))
        samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
        save_images(samples, [image_frame_dim, image_frame_dim],
                    os.path.join(config.sample_dir, 'test_%s.png' % strftime("%Y%m%d%H%M%S", gmtime())))
    # vary single z-component only
    if option == 5:
        values = np.arange(0, 1, 1./config.batch_size)
        for idx in xrange(100):
            print(" [*] %d" % idx)
            z_sample = np.zeros([config.batch_size, dcgan.z_dim])
            for kdx, z in enumerate(z_sample):
                z[idx] = values[kdx]
            save_images(samples, [image_frame_dim, image_frame_dim],
                        os.path.join(config.sample_dir, 'test_arange_%s.png' % (idx)))
