from __future__ import division
import time
import math
import tensorflow as tf
import numpy as np
from six.moves import xrange
from six.moves import urllib
from scipy.ndimage.filters import gaussian_filter

from ops import *
from utils import *
from gauss import gaussian_blur, gauss_kernel_fixed
import hdf5_images

class DCGAN(object):
    def __init__(self, sess, input_height=None, input_width=None, is_crop=False,
                 batch_size=64, sample_num=64, output_height=32, output_width=32,
                 z_dim=100, gf_dim=64, gf_size=5, df_size=5, df_dim=64,
                 gfc_dim=1024, dfc_dim=1024, c_dim=3, dataset_name='default',
                 input_fname_pattern='*data*.pkl', checkpoint_dir=None,
                 num_g_updates=2, data_dir='/home/sagea/scratch/data/icons', is_train=False,
                 label_path=None, gauss_sigma=0, gauss_trunc=2, blur_fade=False, y_dim=None):
        """

        Args:
          sess: TensorFlow session
          batch_size: The size of batch. Should be specified before training.
          z_dim: (optional) Dimension of Z (bottleneck) vector. [100]
          gf_dim: (optional) Number of feature maps in first conv layer for generator. [64]
          df_dim: (optional) Number of feature maps in first conv layer for discrim. [64]
          gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
          dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
          c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
        """
        self.sess = sess
        self.is_crop = is_crop
        self.is_grayscale = (c_dim == 1)

        self.batch_size = batch_size
        self.sample_num = sample_num

        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width

        self.z_dim = z_dim

        self.gf_dim = gf_dim
        self.gf_size = gf_size
        self.df_dim = df_dim
        self.df_size = df_size

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

        self.c_dim = c_dim

        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')

        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')
        self.g_bn3 = batch_norm(name='g_bn3')

        self.dataset_name = dataset_name
        self.input_fname_pattern = input_fname_pattern
        self.checkpoint_dir = checkpoint_dir
        self.data_dir = data_dir
        self.num_g_updates = num_g_updates

        if label_path is None:
            self.has_labels = False
            self.label_path = None
        else:
            self.has_labels = True
            self.label_path = label_path
            self.y_dim = y_dim

        self.gauss_sigma = gauss_sigma
        self.kernel_size = gauss_trunc * 2 + 1
        self.blur_fade = blur_fade

        # --- build model --- #

        # input placeholders
        image_dims = [self.output_height, self.output_width, self.c_dim]
        self.inputs = tf.placeholder(tf.float32, [self.batch_size] + image_dims, name='real_images')
        self.sample_inputs = tf.placeholder(tf.float32, [self.sample_num] + image_dims, name='sample_inputs')
        # latent vector
        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
        self.z_sum = histogram_summary('z', self.z)
        # gauss kernel placeholder
        self.gauss_kernel = tf.placeholder(tf.float32, self.kernel_size, name='gauss_kernel')
        # inputs with added blur
        inputs = gaussian_blur(self.inputs, self.gauss_kernel, self.kernel_size, self.c_dim)
        sample_inputs = gaussian_blur(self.sample_inputs, self.gauss_kernel, self.kernel_size, self.c_dim)

        self.lr = tf.placeholder(tf.float32, shape=[], name='glob_learning_rate')
        self.lr_sum = scalar_summary('glob_learning_rate', self.lr)

        if self.has_labels:
            self.y = tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name='y')
            # generator
            self.G = gaussian_blur(self.generator(self.z, self.y), self.gauss_kernel, self.kernel_size, cdim=c_dim)
            # discriminator for real images
            self.D, self.D_logits = self.discriminator(inputs, self.y)
            # discriminator for generator output
            self.D_, self.D_logits_ = self.discriminator(self.G, self.y, reuse=True)
            # sampler: generator without training batch norm
            self.sampler = gaussian_blur(self.generator(self.z, self.y, sampler=True), self.gauss_kernel,
                                         self.kernel_size, cdim=c_dim)
            # Discriminator sampler assumes unblurred samples
            self.D_sampler, _ = self.discriminator(gaussian_blur(self.inputs, self.gauss_kernel, self.kernel_size, cdim=c_dim),
                                                self.y, sampler=True)
        else:
            # generator
            self.G = gaussian_blur(self.generator(self.z), self.gauss_kernel, self.kernel_size, cdim=c_dim)
            # discriminator for real images
            self.D, self.D_logits = self.discriminator(inputs)
            # discriminator for generator output
            # Todo: understand variable reuse
            self.D_, self.D_logits_ = self.discriminator(self.G, reuse=True)
            # sampler: generator without training batch norm
            self.sampler = self.sampler = gaussian_blur(self.generator(self.z, sampler=True), self.gauss_kernel,
                                                        self.kernel_size, cdim=c_dim)
            self.D_sampler, _ = self.discriminator(gaussian_blur(self.inputs, self.gauss_kernel, self.kernel_size,
                                                              cdim=c_dim), sampler=True)

        self.d_sum = histogram_summary("d", self.D)
        self.d__sum = histogram_summary("d_", self.D_)
        self.G_sum = image_summary("G", self.G)

        def sigmoid_cross_entropy_with_logits(x, y):
            try:
                return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
            except:
                # r0.12
                return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)

        self.d_loss_real = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits, 0.9*tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
        self.g_loss = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))
        self.d_loss = self.d_loss_real + self.d_loss_fake
        self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake)
        self.g_loss_sum = scalar_summary('g_loss', self.g_loss)
        self.d_loss_sum = scalar_summary('d_loss', self.d_loss)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver()

    def z_sampler(self, config):
        """Can be used to get batches of properly distributed latent space samples"""
        if config.sampling == 'uniform':
            z_samples = np.random.uniform(-1, 1, size=(config.batch_size, config.z_dim))
        if config.sampling == 'normal':
            z_samples = np.random.normal(0, 1, size=(config.batch_size, config.z_dim))
        if config.sampling == 't-normal':
            lower, upper = -1, 1
            mu, sigma = 0, config.sigma
            S = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
            z_samples = S.rvs(config.batch_size * config.z_dim).reshape((config.batch_size, config.z_dim))
        if config.sampling == 'sphere':
            z_samples = np.random.normal(0, 1, size=(config.z_dim, config.batch_size))
            z_samples = z_samples / np.linalg.norm(z_samples, axis=0)
            z_samples = np.array([z_samples[:, i] for i in range(config.batch_size)])
        return z_samples.astype(np.float32)

    def discriminator(self, image, y=None, reuse=False, sampler=False):
        """Defines the discriminator CNN architecture"""
        with tf.variable_scope('discriminator') as scope:
            train = True
            if sampler:
                train = False
                reuse = True
            if reuse:
                scope.reuse_variables()
            # Prepare Conditioning vector
            if y is not None:
                yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
                x = conv_cond_concat(image, yb)
            # Discriminator CNN Layers

            if y is None:
                h0 = lrelu(conv2d(image, self.df_dim, k_h=self.df_size, k_w=self.df_size, name='d_h0_conv'))
                h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim * 2, k_h=self.df_size, k_w=self.df_size, name='d_h1_conv'),
                                      train=train))
                h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim * 4, k_h=self.df_size, k_w=self.df_size, name='d_h2_conv'),
                                      train=train))
                h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim * 8, k_h=self.df_size, k_w=self.df_size, name='d_h3_conv'),
                                      train=train))
                h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')
            else:
                yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
                h0 = lrelu(conv2d(conv_cond_concat(image, yb), self.df_dim + self.y_dim,
                                  k_h=self.df_size, k_w=self.df_size, name='d_h0_conv'))
                h1 = lrelu(self.d_bn1(conv2d(conv_cond_concat(h0, yb), self.df_dim * 2 + self.y_dim,
                                      k_h=self.df_size, k_w=self.df_size, name='d_h1_conv'), train=train))
                h2 = lrelu(self.d_bn2(conv2d(conv_cond_concat(h1, yb), self.df_dim * 4 + self.y_dim,
                                      k_h=self.df_size, k_w=self.df_size, name='d_h2_conv'), train=train))
                h3 = lrelu(self.d_bn3(conv2d(conv_cond_concat(h2, yb), self.df_dim * 8 + self.y_dim,
                                      k_h=self.df_size, k_w=self.df_size, name='d_h3_conv'), train=train))
                h3_flat = tf.reshape(h3, [self.batch_size, -1])
                h4 = linear(concat([h3_flat, y], 1), 1, 'd_h3_lin')

            return tf.nn.sigmoid(h4), h4

    def generator(self, z, y=None, sampler=False):
        with tf.variable_scope('generator') as scope:
            train = True
            if sampler:
                train = False
                scope.reuse_variables()
            s_h, s_w = self.output_height, self.output_width
            s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
            s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
            s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
            s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

            # project 'z' and reshape
            if y is not None:
                # print(y)
                z = concat([z, y], 1)
                z_, w_lin, b_lin = linear(z, self.gf_dim * 8 * s_h16 * s_w16, 'g_h0_lin', with_w=True)
                w_y = tf.slice(w_lin, [self.z_dim, 0], [self.y_dim, self.gf_dim * 8 * s_h16 * s_w16])
                self.w_y_sum = tf.summary.histogram('g_label_weigths_histogram', w_y)
            else:
                z_ = linear(z, self.gf_dim * 8 * s_h16 * s_w16, 'g_h0_lin')
            h0 = tf.reshape(z_, [-1, s_h16, s_w16, self.gf_dim * 8])
            h0 = tf.nn.relu(self.g_bn0(h0, train=train))

            if y is None:
                print('no y given')
                h1 = deconv2d(h0, [self.batch_size, s_h8, s_w8, self.gf_dim * 4],
                              k_h=self.gf_size, k_w=self.df_size, name='g_h1')
                h1 = tf.nn.relu(self.g_bn1(h1, train=train))

                h2 = deconv2d(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2],
                              k_h=self.gf_size, k_w=self.df_size, name='g_h2')
                h2 = tf.nn.relu(self.g_bn2(h2, train=train))

                h3 = deconv2d(h2, [self.batch_size, s_h2, s_w2, self.gf_dim],
                              k_h=self.gf_size, k_w=self.df_size, name='g_h3')
                h3 = tf.nn.relu(self.g_bn3(h3, train=train))

                h4 = deconv2d(h3, [self.batch_size, s_h, s_w, self.c_dim],
                              k_h=self.gf_size, k_w=self.df_size, name='g_h4')
            else:
                print('y given')
                yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])

                h1 = deconv2d(conv_cond_concat(h0, yb), [self.batch_size, s_h8, s_w8, self.gf_dim * 4 + self.y_dim],
                              k_h=self.gf_size, k_w=self.df_size, name='g_h1')
                h1 = tf.nn.relu(self.g_bn1(h1, train=train))

                h2 = deconv2d(conv_cond_concat(h1, yb), [self.batch_size, s_h4, s_w4, self.gf_dim * 2 + self.y_dim],
                              k_h=self.gf_size, k_w=self.df_size, name='g_h2')
                h2 = tf.nn.relu(self.g_bn2(h2, train=train))

                h3 = deconv2d(conv_cond_concat(h2, yb), [self.batch_size, s_h2, s_w2, self.gf_dim + self.y_dim],
                              k_h=self.gf_size, k_w=self.df_size, name='g_h3')
                h3 = tf.nn.relu(self.g_bn3(h3, train=train))

                h4 = deconv2d(conv_cond_concat(h3, yb), [self.batch_size, s_h, s_w, self.c_dim],
                              k_h=self.gf_size, k_w=self.df_size, name='g_h4')
            return tf.nn.tanh(h4)

    def train(self, config):
        """Train DCGAN"""

        train_gen, data_len = hdf5_images.load(batch_size=self.batch_size, data_file=self.data_dir,
                                               resolution=self.output_height, label_name=self.label_path)
        def inf_train_gen():
            while True:
                for _images, _labels in train_gen():
                    yield _images, _labels
        gen = inf_train_gen()

        d_optim = tf.train.AdamOptimizer(self.lr, beta1=config.beta1) \
            .minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(self.lr, beta1=config.beta1) \
            .minimize(self.g_loss, var_list=self.g_vars)

        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_tables().run()
        self.g_sum = merge_summary([self.G_sum, self.d_loss_fake_sum, self.g_loss_sum]) #self.z_sum, self.d__sum (hist)
        self.d_sum = merge_summary([self.lr_sum, self.z_sum, self.d_loss_real_sum, self.d_loss_sum]) #self.d_sum (hist)
        # label weight histogram
        # if self.has_labels:
        #     self.g_sum = merge_summary([self.g_sum, self.w_y_sum])

        # initialize summary writer: for each run create (enumerated) sub-directory
        if os.path.exists('./logs/' + self.dataset_name):
            # number of existing immediate child directories of log folder
            run_var = len(next(os.walk('./logs/' + self.dataset_name))[1]) + 1
        else:
            run_var = 1
        self.writer = SummaryWriter('%s/%s' % ('./logs/' + self.dataset_name, run_var), self.sess.graph)

        sample_z = self.z_sampler(config)
        sample_inputs = gen.next()
        sample_images = sample_inputs[0]
        if self.has_labels:
            # use one of the following two lines to get either random samples or only samples from the first 8 classes
            # sample_y = np.eye(self.y_dim)[sample_inputs[1]]
            sample_y = np.eye(self.y_dim)[[1, 2, 3, 4, 5, 6, 7, 8] * 8]

        counter = 1
        start_time = time.time()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        batch_idxs = data_len // self.batch_size
        for epoch in xrange(config.epoch):
            if self.blur_fade:
                # sigma_used = self.gauss_sigma * (1 - (epoch / config.epoch))
                # tapered version
                full_blur_ep = 0 # int(config.epoch * 0.2)
                no_blur_ep = 6 # int(config.epoch * 0.2)
                fade_ep = config.epoch - full_blur_ep - no_blur_ep
                if epoch < full_blur_ep:
                    sigma_used = self.gauss_sigma
                elif epoch < full_blur_ep + fade_ep:
                    sigma_used = self.gauss_sigma * (1 - ((epoch + 1 - full_blur_ep) / fade_ep))
                else:
                    sigma_used = 0
            else:
                sigma_used = self.gauss_sigma
            kernel_used = gauss_kernel_fixed(sigma_used, (self.kernel_size - 1) // 2)
            for idx in xrange(0, batch_idxs):
                batch_z = self.z_sampler(config)
                # when using random labels (commented-out line below), the fake label distribution
                # might not match the data and training can fail!
                # batch_y = tf.one_hot(np.random.random_integers(0, self.y_dim - 1, config.batch_size), self.y_dim)
                # better use the real images and labels:
                batch_images, batch_labels_num = gen.next()
                # if images are stored in BHWC-format, the following line should be commented out
                batch_images = batch_images.transpose((0, 2, 3, 1))
                if self.has_labels:
                    batch_labels = np.eye(self.y_dim)[batch_labels_num]
                if config.blur_input is not None:
                    batch_images = gaussian_filter(batch_images, [0, config.blur_input, config.blur_input, 0])

                # Update D network
                _, summary_str = self.sess.run([d_optim, self.d_sum],
                                               feed_dict={self.inputs: batch_images, self.z: batch_z,  self.lr: config.learning_rate,
                                                          self.gauss_kernel: kernel_used} if not self.has_labels
                                               else {self.inputs: batch_images, self.y: batch_labels, self.lr: config.learning_rate,
                                                     self.z: batch_z, self.gauss_kernel: kernel_used})
                self.writer.add_summary(summary_str, counter)

                # Update G network specified number of times
                for i in range(self.num_g_updates):
                    _, summary_str = self.sess.run([g_optim, self.g_sum],
                                                   feed_dict={self.inputs: batch_images, self.lr: config.learning_rate,
                                                              self.z: batch_z, self.gauss_kernel: kernel_used} if not self.has_labels
                                                   else {self.inputs: batch_images, self.y: batch_labels, self.lr: config.learning_rate,
                                                         self.z: batch_z, self.gauss_kernel: kernel_used})
                    self.writer.add_summary(summary_str, counter)

                # Get losses for current batch
                errD_fake = self.d_loss_fake.eval({self.z: batch_z, self.gauss_kernel: kernel_used} if not self.has_labels
                                                  else {self.z: batch_z, self.y: batch_labels, self.gauss_kernel: kernel_used})
                errD_real = self.d_loss_real.eval({self.inputs: batch_images, self.gauss_kernel: kernel_used} if not self.has_labels
                                                  else {self.inputs: batch_images, self.y: batch_labels, self.gauss_kernel: kernel_used})
                errG = self.g_loss.eval({self.z: batch_z, self.gauss_kernel: kernel_used} if not self.has_labels
                                        else {self.z: batch_z, self.y: batch_labels, self.gauss_kernel: kernel_used})

                # add sigma to summary
                sigma_sum = tf.Summary(value=[tf.Summary.Value(tag='gauss_sigma', simple_value=sigma_used)])
                self.writer.add_summary(sigma_sum, counter)

                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f, sigma: %4f"
                      % (epoch, idx, batch_idxs, time.time() - start_time, errD_fake+errD_real, errG, sigma_used))
                # print losses for constant sample and save image with corresponding output
                if (counter % 500) == 1:
                    try:
                        samples, d_loss, g_loss = self.sess.run(
                            [self.sampler, self.d_loss, self.g_loss],
                            feed_dict={self.z: sample_z, self.inputs: sample_images,
                                       self.y: sample_y, self.gauss_kernel: kernel_used} if self.has_labels
                            else {self.z: sample_z, self.inputs: sample_images, self.gauss_kernel: kernel_used})
                        manifold_h = int(np.ceil(np.sqrt(samples.shape[0])))
                        manifold_w = int(np.floor(np.sqrt(samples.shape[0])))
                        save_images(samples, [manifold_h, manifold_w],
                                    './{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))
                        print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))
                    except Exception, e:
                        print("Error saving sample image:")
                        print e
                # save checkpoint
                if np.mod(counter, 3000) == 2:
                    self.save(config.checkpoint_dir, counter)

    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(
            self.dataset_name, self.batch_size,
            self.output_height, self.output_width)

    def save(self, checkpoint_dir, step):
        model_name = "DCGAN.model"
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0
