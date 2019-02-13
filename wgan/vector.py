from __future__ import division
import os
import math
import scipy.misc
import numpy as np

import tensorflow as tf
import tensorflow.contrib.slim as slim

import tflib.inception_score
from tflib.utils import *
from tqdm import tqdm

from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
import h5py

from scipy import signal
from scipy.ndimage.filters import convolve


class Vector(object):
    def __init__(self, wgan):
        self.wgan = wgan
        self.cfg = wgan.cfg
        self.const_y = None
        self.z_dim = 128

    def show_random(self, shape=(), border=True, enum=False, res=32, ret_vec=False, save=None):
        """
        Shows a collection of  random samples with a pre-determined shape
        If shape is an empty tuple, a standard batch is shown, if it is a 1-tuple it defines
        the width of the shown image instead of a square
        :param shape: Tuple defining the shape (in number of samples) of the output.
        :param border: Flag, should there be a white border in-between samples?
        :param enum: Flag, should the samples be enumerated?
        :param res: Output resolution of a single sample in pixels (assumed square)
        :param ret_vec: Flag, should the latent vecotr and labels be returned?
        :param save: Path to image file if output should be saved, 'None' if not
        :return: Returns (z, y) if ret_vec==True
        """
        """"""
        if len(shape) <= 1:
            size = self.cfg.BATCH_SIZE
        elif len(shape) == 2:
            size = shape[0] * shape[1]
        else:
            raise Exception('Invalid shape specified: %s' % str(shape))
        z = self.gen_z(size)
        y = self.gen_y(size=size)
        self.show_z(z, y, shape=shape, border=border, enum=enum, res=res, save=save)
        if ret_vec:
            return z, y

    def gen_z(self, size=None):
        if size is None:
            return self.wgan.z_sampler()
        ret = []
        n_batches = size // self.cfg.BATCH_SIZE + (1 if size % self.cfg.BATCH_SIZE else 0)
        for i in range(n_batches):
            ret.append(self.wgan.z_sampler())
        ret = np.concatenate(ret)
        return ret[:size]

    def gen_y(self, number=None, size=None):
        if size is None:
            size = self.cfg.BATCH_SIZE
        if number is None:
            if self.cfg.N_LABELS > 0:
                with h5py.File(self.cfg.DATA) as hf:
                    probs = hf[self.cfg.LABELS].attrs['probs']
                number = np.random.choice(range(self.cfg.N_LABELS), size=size, replace=True, p=probs)
                # number = np.random.randint(0, self.cfg.N_LABELS, size)
            else:
                number = np.array([0]*size)
        if type(number) is int:
            number_array = np.full(size, number, dtype=np.int32)
        else:
            number_array = np.array(number)
        return np.eye(self.cfg.N_LABELS)[number_array] if self.cfg.LAYER_COND else number_array

    def set_y(self, number):
        self.const_y = self.gen_y(number, size=self.cfg.BATCH_SIZE)

    def sample_z(self, z, y=None):
        if y is None:
            if self.const_y is None:
                raise Exception('No constant label set, please set first or call this function with parameter y')
            else:
                y = self.const_y
        if len(z.shape) == 1:
            z = np.reshape(z, (1, len(z)))
        # todo: implement some label format parameter
        y_dim = (self.cfg.N_LABELS if self.cfg.LAYER_COND else 1)
        ret = []
        n_batches = len(z) // self.cfg.BATCH_SIZE + (1 if len(z) % self.cfg.BATCH_SIZE else 0)
        for i in range(n_batches):
            z_i = z[self.cfg.BATCH_SIZE * i:self.cfg.BATCH_SIZE * (i + 1)]
            if len(y) > self.cfg.BATCH_SIZE:
                y_i = y[self.cfg.BATCH_SIZE * i:self.cfg.BATCH_SIZE * (i + 1)]
            else:
                y_i = y
            pad = self.cfg.BATCH_SIZE - len(z_i)
            if pad > 0:
                # todo: replace zero-padding with samples --> messes up batchnorm!
                z_i = np.concatenate((z_i, np.zeros((pad, self.z_dim), dtype=np.float32)))
                if y_i is not self.const_y:
                    y_i = np.concatenate((y_i, np.zeros((pad, y_dim) if y_dim > 1 else pad, dtype=np.float32)))
            samples = self.wgan.sample(z_i, y_i)
            ret.append(samples[:self.cfg.BATCH_SIZE - pad])
        return np.concatenate(ret)

    def show(self, images, shape=(), enum=False, border=False, transform=False, res=32, save=False):
        # shape follows PIL convention (width, height)
        if len(images.shape) <= 3:
            img = scipy.misc.toimage(images)
            img = img.resize((res, res), resample=Image.LANCZOS)
            img.show()
        else:
            if len(shape) == 1:
                shape = (shape, int(math.ceil(len(images) / shape)))
            elif len(shape) == 0:
                grid_size = int(math.ceil(math.sqrt(len(images))))
                shape = (grid_size, grid_size)
            if type(border) is int:
                border_width = border
            else:
                border_width = 10 if res == 32 else 20
            if enum:
                images = np.array([self.add_no(img, n, border, transform) for n, img in enumerate(images)])
                if border:
                    image_size_v = shape[0] * (res + border_width)
                image_size_h = shape[1] * (res + border_width)
            elif border:
                images = np.array([self.add_no(img, None, border, transform) for img in images])
                image_size_v = shape[0] * (res + border_width)
                image_size_h = shape[1] * (res + border_width)
            else:
                image_size_h = shape[1] * res
                image_size_v = shape[0] * res
            merged = merge(images, (shape[1], shape[0]))
            img = scipy.misc.toimage(merged)
            img = img.resize((image_size_v, image_size_h), resample=Image.LANCZOS)
            if not save:
                img.show()
            else:
                with open(save, 'wb') as f:
                    img.save(f, format='png')

    def show_z(self, z, y=None, shape=(), enum=True, border=True, res=32, save=False):
        self.show(self.sample_z(z, y), shape=shape, enum=enum, border=border, res=res, save=save)

    @staticmethod
    def avg(z_samples):
        return np.average(z_samples, axis=0)

    def add_no(self, image, number, border=True, transform=False):
        """ Adds a number to the bottom of each generated sample as well as an optional border in-between"""
        res = image.shape[0]
        if transform:
            image = ((image + 1) / 2 * 255).astype(np.uint8)
        if res == 32:
            f_size, i_size = 8, (32, 10)
        elif res == 64:
            f_size, i_size = 16, (64, 20)
        else:
            raise Exception("unsupported image size")
        if number is not None:
            font = ImageFont.truetype("/usr/share/fonts/truetype/ttf-dejavu/DejaVuSans-Bold.ttf", f_size)
            text = str(number)
            tcolor = (0, 0, 0)
            text_pos = (0, 0)
        img = Image.new('RGB', i_size, color=(255, 255, 255))
        if number is not None:
            draw = ImageDraw.Draw(img)
            draw.text(text_pos, text, fill=tcolor, font=font)
        arr = np.array(img)
        if border:
            bd = Image.new('RGB', (i_size[1] // 2, i_size[0] + i_size[1]), color=(255, 255, 255))
            bd_arr = np.array(bd)
            return np.hstack((bd_arr, np.vstack((image, arr)), bd_arr))
        return np.vstack((image, arr))

    def interpolate(self, z_start, z_stop, y_start=None, y_stop=None, n_steps=62, method='gaussian-matched'):
        """interpolates from vector start to stop in n steps, using the specified method
            one-hot encoded labels are always interpolated linearly"""
        steps = np.linspace(0, 1, n_steps + 2)
        z_samples = np.array([self.intermediate(step, z_start, z_stop, method) for step in steps])
        if y_start is None:
            samples = self.sample_z(z_samples)
        else:
            y_samples = np.array([self.intermediate(step, y_start, y_stop, 'linear') for step in steps])
            samples = self.sample_z(z_samples, y_samples)
        return samples

    def intermediate(self, pos, start, stop, method='gaussian-matched'):
        """calculates the intermediate vector between start and stop using the method specified"""
        if method == 'slerp':
            return slerp(pos, start, stop)
        elif method == 'linear':
            return start + pos * (stop - start)
        # elif method == 'uniform-matched':
        #     if pos == 0:
        #         return start
        #     if pos == 1:
        #         return stop
        #     y_lin = start + pos * (stop - start)
        #     return map(lambda y: Ytilde(y, pos), y_lin)
        elif method == 'gaussian-matched':
            if pos == 0:
                return start
            if pos == 1:
                return stop
            return (start + pos * (stop - start)) / math.sqrt(pos ** 2 + (1 - pos) ** 2)
        else:
            raise ValueError("Unrecognized interpolation method specified!")

    def ip_show(self, z_start, z_stop, y_start=None, y_stop=None, shape=(), method='gaussian-matched',
                enum=True, border=True, transform=False, res=32):
        if len(shape) <= 1:
            n_steps = self.cfg.BATCH_SIZE - 2
        elif len(shape) == 2:
            n_steps = (shape[0] * shape[1]) - 2
        else:
            raise ValueError("Shape paramteter has an unsupported shape")
        ip_z = self.interpolate(z_start, z_stop, y_start, y_stop, n_steps, method)
        self.show(ip_z, shape=shape, enum=enum, border=border, transform=transform, res=res)

    def get_inception_score(self, all_samples, splits=10):
        return tflib.inception_score.get_inception_score(list(all_samples), splits=splits)

    def data_score(self, n, res=None):
        if res is None:
            res = self.cfg.OUTPUT_RES

        if self.cfg.DATA_LOADER == 'hdf5':
            with h5py.File(self.cfg.DATA, 'r') as h_f:
                all_samples = h_f['data'][:n]
        elif self.cfg.DATA_LOADER == 'twitter':
            data_loader = self.wgan.get_data_loader()
            train_gen = data_loader.load_new(self.wgan.cfg)

            def inf_train_gen():
                while True:
                    for _, labels in train_gen():
                        yield labels

            all_samples = []
            for i in tqdm(xrange(n // self.wgan.cfg.BATCH_SIZE)):
                all_samples.append(self.sample_z(z=self.gen_z(), y=inf_train_gen().next()))
            all_samples = np.concatenate(all_samples, axis=0)
        return self.get_inception_score(list(all_samples.transpose((0, 2, 3, 1))))

    def score(self, n, resize=0, ret_samples=False):
        data_loader = self.wgan.get_data_loader()
        train_gen = data_loader.load_new(self.wgan.cfg)

        def inf_train_gen():
            while True:
                for _, labels in train_gen():
                    yield labels
        all_samples = []
        for i in tqdm(xrange(n // self.wgan.cfg.BATCH_SIZE)):
            all_samples.append(self.sample_z(z=self.gen_z(), y=inf_train_gen().next()))
        all_samples = np.concatenate(all_samples, axis=0)
        if resize:
            all_samples = np.array([scipy.misc.imresize(sample, (resize, resize)) for sample in tqdm(all_samples)])
        if ret_samples:
            return all_samples
        else:
            return self.get_inception_score(list(all_samples))
