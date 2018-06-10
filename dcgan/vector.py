from __future__ import division
import scipy.misc
import numpy as np
import h5py

from ops import *
from utils import *
from transform import Ytilde
from tqdm import tqdm
import hdf5_images
import inception_score

from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw


class Vector(object):
    def __init__(self, sess, dcgan, config):
        self.sess = sess
        self.dcgan = dcgan
        self.config = config
        self.gauss_kernel = gauss_kernel_fixed(config.gauss_sigma, config.gauss_trunc)
        self.const_y = None

    def show_random(self, shape=(), border=True, enum=False, res=32, ret_vec=False):
        """Shows random samples of shape 'shape'
        If shape is an empty tuple, a standard batch is shown, if it is a 1-tuple it defines
        the width of the shown image instead of a square"""
        if len(shape) <= 1:
            size = self.dcgan.batch_size
        elif len(shape) == 2:
            size = shape[0] * shape[1]
        z = self.gen_z(size)
        if self.dcgan.has_labels:
            y = self.gen_y(size=size)
        else:
            y = None
        self.show_z(z, y, shape=shape, border=border, enum=enum, res=res)
        if ret_vec:
            return z, y

    def gen_z(self, size=None):
        if size is None:
            return self.dcgan.z_sampler(self.config)
        ret = []
        n_batches = size // self.dcgan.batch_size + (1 if size % self.dcgan.batch_size else 0)
        for i in range(n_batches):
            ret.append(self.dcgan.z_sampler(self.config))
        ret = np.concatenate(ret)
        return ret[:size]

    def gen_y(self, number=None, size=None):
        if size is None:
            size = self.dcgan.batch_size
        if number is None:
            with h5py.File(self.config.data_dir) as hf:
                probs = hf[self.config.label_file].attrs['probs']
            number = np.random.choice(range(self.dcgan.y_dim), size=size, replace=True, p=probs)
            # number = np.random.randint(0, self.dcgan.y_dim, size)
        if size == 1:
            return np.eye(self.dcgan.y_dim)[number]
        else:
            if type(number) is int:
                number_array = np.full(size, number, dtype=np.int32)
            else:
                number_array = np.array(number)
            return np.eye(self.dcgan.y_dim)[number_array]

    def set_y(self, number):
        self.const_y = self.gen_y(number, size=self.dcgan.batch_size)

    def sample_z(self, z, y=None):
        if y is None and self.dcgan.has_labels:
            if self.const_y is None:
                raise Exception('No constant label set, please set first or call this function with parameter y')
            else:
                y = self.const_y
        if len(z.shape) == 1:
            z = np.reshape(z, (1, len(z)))
        ret = []
        n_batches = len(z) // self.dcgan.batch_size + (1 if len(z) % self.dcgan.batch_size else 0)
        for i in range(n_batches):
            z_i = z[self.dcgan.batch_size * i:self.dcgan.batch_size * (i + 1)]
            if self.dcgan.has_labels:
                y_i = y[self.dcgan.batch_size * i:self.dcgan.batch_size * (i + 1)]
            pad = self.dcgan.batch_size - len(z_i)
            if pad > 0:
                z_i = np.concatenate((z_i, np.zeros((pad, self.dcgan.z_dim), dtype=np.float32)))
                if self.dcgan.has_labels and (len(y_i) < self.dcgan.batch_size):
                    y_i = np.concatenate((y_i, np.zeros((pad, self.dcgan.y_dim), dtype=np.float32)))
            if self.dcgan.has_labels:
                samples = self.sess.run(self.dcgan.sampler, feed_dict={self.dcgan.z: z_i, self.dcgan.y: y_i,
                                                                       self.dcgan.gauss_kernel: self.gauss_kernel})
            else:
                samples = self.sess.run(self.dcgan.sampler, feed_dict={self.dcgan.z: z_i,
                                                                       self.dcgan.gauss_kernel: self.gauss_kernel})
            ret.append(samples[:self.dcgan.batch_size - pad])
        return np.concatenate(ret)

    def show(self, images, shape=(), enum=False, border=False, transform=True, res=32, save=False):
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

    def add_no(self, image, number, border=True, transform=True):
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

    def interpolate(self, z_start, z_stop, y_start=None, y_stop=None, n_steps=62, method='slerp'):
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

    def intermediate(self, pos, start, stop, method='slerp'):
        """calculates the intermediate vector between start and stop using the method specified"""
        if method == 'slerp':
            return slerp(pos, start, stop)
        elif method == 'linear':
            return start + pos * (stop - start)
        elif method == 'uniform-matched':
            if pos == 0:
                return start
            if pos == 1:
                return stop
            y_lin = start + pos * (stop - start)
            return map(lambda y: Ytilde(y, pos), y_lin)
        elif method == 'gaussian-matched':
            return (start + pos * (stop - start)) / math.sqrt(pos ** 2 + (1 - pos) ** 2)

    def ip_show(self, z_start, z_stop, y_start=None, y_stop=None, shape=(), method='slerp',
                enum=True, border=True, transform=True, res=32):
        if len(shape) <= 1:
            n_steps = self.dcgan.batch_size - 2
        elif len(shape) == 2:
            n_steps = (shape[0] * shape[1]) - 2
        ip_z = self.interpolate(z_start, z_stop, y_start, y_stop, n_steps, method)
        self.show(ip_z, shape=shape, enum=enum, border=border, transform=transform, res=res)

    def get_d(self, image, y=None):
        if y is None and self.dcgan.has_labels:
            if self.const_y is None:
                raise Exception('No constant label set, please set first or call this function with parameter y')
            else:
                y = self.const_y
        if len(image.shape) == 3:
            image = np.reshape(image, [1] + image.shape)
        ret = []
        n_batches = len(image) // self.dcgan.batch_size + (1 if len(image) % self.dcgan.batch_size else 0)
        for i in range(n_batches):
            image_i = image[self.dcgan.batch_size * i:self.dcgan.batch_size * (i + 1)]
            if self.dcgan.has_labels:
                y_i = y[self.dcgan.batch_size * i:self.dcgan.batch_size * (i + 1)]
            pad = self.dcgan.batch_size - len(image_i)
            if pad > 0:
                image_i = np.concatenate((image_i, np.zeros((pad,) + image.shape[1:], dtype=np.float32)))
                if self.dcgan.has_labels and y_i is not self.const_y:
                    y_i = np.concatenate((y_i, np.zeros((pad, self.dcgan.y_dim), dtype=np.float32)))
            if self.dcgan.has_labels:
                samples = self.sess.run(self.dcgan.D_sampler, feed_dict={self.dcgan.inputs: image_i, self.dcgan.y: y_i,
                                                                         self.dcgan.gauss_kernel: self.gauss_kernel})
            else:
                samples = self.sess.run(self.dcgan.D_sampler, feed_dict={self.dcgan.inputs: image_i,
                                                                         self.dcgan.gauss_kernel: self.gauss_kernel})
            ret.append(samples[:self.dcgan.batch_size - pad])
        return np.concatenate(ret)

    def get_inception_score(self, n, resize=0, ret_samples=False):
        all_samples = []
        if self.dcgan.has_labels:
            train_gen, _ = hdf5_images.load(batch_size=self.dcgan.batch_size, data_file=self.dcgan.data_dir,
                                            resolution=self.dcgan.output_height, label_name=self.dcgan.label_file)

            def inf_train_gen():
                while True:
                    for _, labels in train_gen():
                        yield labels
        for i in tqdm(xrange(n // self.dcgan.batch_size)):
            if self.dcgan.has_labels:
                all_samples.append(self.sample_z(z=self.gen_z(), y=np.eye(self.dcgan.y_dim)[inf_train_gen().next()]))
            else:
                all_samples.append(self.sample_z(z=self.gen_z()))
        all_samples = np.concatenate(all_samples, axis=0)
        if resize:
            all_samples = np.array([scipy.misc.imresize(sample, (resize, resize)) for sample in tqdm(all_samples)])
            all_samples = (all_samples + 1) * 127.5
        if ret_samples:
            return all_samples
        return inception_score.get_inception_score(list(all_samples))
