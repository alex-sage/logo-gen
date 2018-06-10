"""Improved Wasserstein GAN code for the CVPR paper
   "Logo Synthesis and Manipulation with Clustered Generative Adversarial Networks"
   Based in large parts on the official codebase for "Improved Training of Wasserstein GANs" by Gulrajani et al.
   Available at https://github.com/igul222/improved_wgan_training"""

import os
import argparse

import tflib as lib
import tflib.save_images
import tflib.cifar10
import tflib.twitter_images
import tflib.hdf5_images
import tflib.small_imagenet
import tflib.inception_score
import tflib.architectures

import numpy as np
import tensorflow as tf
import json
from shutil import copyfile

import time
import locale

locale.setlocale(locale.LC_ALL, '')


class Config(object):
    """Config class used to set all the parameter used for GAN training and inference.
       It can be used by passing an instantiated object of the script to the training function.
       Alternatively, all parameters can aso be passed as named parameters on the command line."""

    def __init__(self, initial_data=None, **kwargs):
        # default values
        self.DATA_LOADER = 'hdf5'  # Data format to be used
        self.DATA = 'data/LLD-icon.hdf5'  # Path to training data (folder or file depending on fromat)
        self.LABELS = 'labels/resnet/rc_128'  # Path to labels: Either the filesystem location of a pickle file
        #                                      containing the labels or the path to the label dataset within a HDF5 file
        self.N_LABELS = 0  # Number of label classes
        self.RUN_NAME = 'layer_cond_clean'  # name for this experiment run
        self.N_GPUS = 1
        self.ARCHITECTURE = 'resnet-32'  # used GAN architecture
        self.MODE = 'wgan-gp'  # training mode
        self.BATCH_SIZE = 64  # Critic batch size
        self.GEN_BS_MULTIPLE = 2  # Generator batch size, as a multiple of BATCH_SIZE
        self.ITERS = 100000  # How many iterations to train for
        self.DIM_G = 128  # Generator dimensionality
        self.DIM_D = 128  # Critic dimensionality
        self.NORMALIZATION_G = 1  # BOOLEAN Use batchnorm in generator?
        self.NORMALIZATION_D = 0  # BOOLEAN Use batchnorm (or layernorm) in critic?
        self.OUTPUT_RES = 32
        self.LR = 0  # Initial learning rate [0 --> default]
        self.LAMBDA = 10  # gradient penalty lambda

        self.DECAY = 1  # BOOLEAN Whether to decay LR over learning
        self.N_CRITIC = 5  # Critic steps per generator steps (except for lsgan and DCGAN training modes)
        self.N_GENERATOR = 3  # Generator steps per critic step for DCGAN training mode
        self.INCEPTION_FREQUENCY = 0  # How frequently to calculate Inception score
        self.SUMMARY_FREQUENCY = 1  # How frequently to write out a tensorboard summary
        self.KEEP_CHECKPOINTS = 5  # Number of checkpoints to keep (long-term, spread out over entire training time)

        self.CONDITIONAL = 0  # BOOLEAN Whether to train a conditional or unconditional model
        self.LAYER_COND = 0  # BOOLEAN feed the labels to every layer in generator and discriminator
        self.ACGAN = 0  # BOOLEAN If CONDITIONAL, whether to use ACGAN or "vanilla" conditioning
        self.ACGAN_SCALE = 1.  # How to scale the critic's ACGAN loss relative to WGAN loss
        self.ACGAN_SCALE_G = 0.1  # How to scale generator's ACGAN loss relative to WGAN loss

        self.bn_init = False

        if self.N_GPUS not in [1, 2]:
            raise Exception('Only 1 or 2 GPUs supported!')
        if len(self.DATA) == 0:
            raise Exception('Please specify path to data directory in gan_cifar.py!')
        if self.LR == 0:
            if self.MODE == 'wgan-gp':
                if self.ARCHITECTURE == 'resnet-32':
                    self.LR = 2e-4
                else:
                    self.LR = 1e-4
            if self.MODE == 'wgan':
                self.LR = 5e-5
            if self.MODE == 'dcgan':
                self.LR = 2e-4
            if self.MODE == 'lsgan':
                self.LR = 1e-4
            if self.MODE == 'ae-l2':
                self.LR = 2e-4

        # read config from dict or keywords
        if initial_data is not None:
            for key in initial_data:
                setattr(self, key, initial_data[key])
        for key in kwargs:
            setattr(self, key, kwargs[key])

        # derived parameterseither
        self.OUTPUT_DIM = self.OUTPUT_RES * self.OUTPUT_RES * 3  # Number of pixels in each image

    def __str__(self):
        return str(self.__dict__)

    def __cmp__(self, other):
        return self.__dict__ == other.__dict__


class WGAN(object):
    """This class contains all functions used for GAN training and inference.
       An object can be instantiated by loading the parameters from file, passing them as a dict or as named parameters.
       session:     TF session
       load_config: Path to the config.json file containing the parameters (typically of a previously trained GAN)
       config_dict: Dictionary representation a Config object
       **kwargs:    Additional named parameters (will override those loaded from file or dict)"""

    def __init__(self, session, load_config=None, config_dict=None, **kwargs):
        self.session = session
        self.sampler_initialized = 0
        self.current_iter = 0
        self.update_moving_stats = False
        self.t_train = tf.placeholder(tf.bool)
        # create config object
        if load_config is not None:
            with open(os.path.join('runs', load_config, 'config.json'), 'r') as f:
                loaded_dict = json.load(f)
            cfg = Config(loaded_dict, **kwargs)
        else:
            cfg = Config(config_dict, **kwargs)
        self.cfg = cfg

        if cfg.CONDITIONAL and (not cfg.ACGAN) and (not cfg.LAYER_COND) and (not cfg.NORMALIZATION_D):
            print "WARNING! Conditional model without normalization in D might be effectively unconditional!"

        # returns a (Generator, Discriminator) pair according to config
        def get_architecture(cfg):
            if cfg.ARCHITECTURE == 'dcgan-32':
                return tflib.architectures.Generator_DCGAN_32, tflib.architectures.Discriminator_DCGAN_32
            if cfg.ARCHITECTURE == 'dcgan-64':
                return tflib.architectures.Generator_DCGAN_64, tflib.architectures.Discriminator_DCGAN_64
            if cfg.ARCHITECTURE == 'm-dcgan-64':
                return tflib.architectures.Generator_MultiplicativeDCGAN_64, \
                       tflib.architectures.Discriminator_MultiplicativeDCGAN_64
            if cfg.ARCHITECTURE == 'resnet-32':
                return tflib.architectures.Generator_Resnet_32, tflib.architectures.Discriminator_Resnet_32
            if cfg.ARCHITECTURE == 'resnet-64':
                return tflib.architectures.Generator_Resnet_64, tflib.architectures.Discriminator_Resnet_64
            if cfg.ARCHITECTURE == 'resnet-128':
                return tflib.architectures.Generator_Resnet_128, tflib.architectures.Discriminator_Resnet_128
            if cfg.ARCHITECTURE == 'b-resnet-64':
                return tflib.architectures.Generator_Bottleneck_Resnet_64, \
                       tflib.architectures.Discriminator_Bottleneck_Resnet_64
            if cfg.ARCHITECTURE == 'fc-64':
                return tflib.architectures.Generator_FC_64, tflib.architectures.Discriminator_FC_64

        self.Generator, self.Discriminator = get_architecture(cfg)

        lib.print_model_settings_dict(cfg.__dict__)

        self.run_dir = os.path.join('runs', cfg.RUN_NAME)
        self.save_dir = os.path.join(self.run_dir, 'checkpoints')
        self.sample_dir = os.path.join(self.run_dir, 'samples')
        self.tb_dir = os.path.join(self.run_dir, 'tensorboard')
        self.saver = None

        def maybe_mkdirs(path):
            if type(path) is list:
                for path in path:
                    maybe_mkdirs(path)
            else:
                if not os.path.exists(path):
                    os.makedirs(path)

        maybe_mkdirs([self.run_dir, self.save_dir, self.sample_dir, self.tb_dir])

        self._iteration = tf.placeholder(tf.int32, shape=None)
        self.all_real_data_int = tf.placeholder(tf.int32, shape=[cfg.BATCH_SIZE, 3, cfg.OUTPUT_RES, cfg.OUTPUT_RES])
        self.all_real_labels = tf.placeholder(tf.int32, shape=[cfg.BATCH_SIZE])
        self.fixed_noise = tf.constant(np.random.normal(size=(100, 128)).astype('float32'))
        if cfg.LAYER_COND:
            self.fixed_labels = tf.one_hot(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9] * 10, dtype='int32'),
                                           depth=cfg.N_LABELS)
        else:
            self.fixed_labels = tf.constant(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9] * 10, dtype='int32'))
        self.z = None
        self.y = None
        self.label_probs = None
        # if cfg.DATA_LOADER in ['hdf5', 'twitter'] and cfg.LABELS != 'None':
        #     with h5py.File(cfg.DATA, 'r') as f:
        #         cfg.label_probs = f[cfg.LABELS].attrs['probs']

        ## copy data to scratch, only works for single files!
        if (cfg.DATA_LOADER == 'hdf5' or cfg.DATA_LOADER == 'twitter') and not os.path.exists(self.cfg.DATA):
            if not os.path.exists(os.path.dirname(self.cfg.DATA)):
                os.makedirs(os.path.dirname(self.cfg.DATA))
            source = ['', 'home'] + [self.cfg.DATA.split('/')[2]] + ['scratch'] + self.cfg.DATA.split('/')[3:]
            copyfile('/'.join(source), self.cfg.DATA)
            print('copied data')

    def get_data_loader(self):
        if self.cfg.DATA_LOADER == 'pickle':
            return lib.batchloader
        if self.cfg.DATA_LOADER == 'cifar-10':
            return lib.cifar10
        if self.cfg.DATA_LOADER == 'lld-logo':
            return lib.twitter_images
        if self.cfg.DATA_LOADER == 'hdf5':
            return lib.hdf5_images
        if self.cfg.DATA_LOADER == 'imagenet-small':
            return lib.small_imagenet

    # restores a GAN model from checkpoint
    def restore_model(self):
        # initialize saver
        if self.saver is None:
            self.saver = tf.train.Saver()
        # try to restore checkpoint
        ckpt = tf.train.get_checkpoint_state(self.save_dir)
        if ckpt:
            with open(os.path.join(self.run_dir, 'config.json'), 'r') as f:
                old_dict = json.load(f)
            new_dict = self.cfg.__dict__
            equal = True
            for key, value in old_dict.iteritems():
                if (key != 'train') and (key[:3] != 'bn_') and new_dict[key] != value:
                    print('New: %s: %s' % (key, new_dict[key]))
                    print('Old: %s: %s' % (key, value))
                    equal = False
            if not equal:
                raise Exception('Config for existing checkpoint is not the same, aborting!')
            self.saver.restore(self.session, ckpt.model_checkpoint_path)
            self.current_iter = int(np.loadtxt(os.path.join(self.save_dir, 'last_iteration')))
            print('model restored.')
        else:
            with open(os.path.join(self.run_dir, 'config.json'), 'w') as f:
                json.dump(self.cfg.__dict__, f)

    # initializes the sampler (used for GAN inference)
    def _init_sampler(self):
        t_False = tf.constant(True, dtype=tf.bool)
        self.update_moving_stats = False
        y_shape = (None, self.cfg.N_LABELS) if self.cfg.LAYER_COND else (None,)
        self.z = tf.placeholder(tf.float32, shape=(None, 128), name='z')
        self.y = tf.placeholder((tf.float32 if self.cfg.LAYER_COND else tf.int32), shape=y_shape, name='y')
        all_real_data = tf.reshape(2 * ((tf.cast(self.all_real_data_int, tf.float32) / 256.) - .5),
                                   [self.cfg.BATCH_SIZE, self.cfg.OUTPUT_DIM])
        all_real_data += tf.random_uniform(shape=[self.cfg.BATCH_SIZE, self.cfg.OUTPUT_DIM], minval=0.,
                                           maxval=1. / 128)  # dequantize
        self.sampler = self.Generator(self.cfg, n_samples=0, labels=self.y, noise=self.z, is_training=self.t_train)
        # if sampler_d not needed, is_training parameter can be removed
        self.sampler_d = self.Discriminator(self.cfg, inputs=all_real_data, labels=self.y, is_training=self.t_train)
        self.restore_model()
        self.sampler_initialized = True

    # returns a batch of latent space samples with the correct distribution
    def sample(self, z=None, y=None):
        if z is None:
            z = np.random.normal(size=(100, 128)).astype('float32')
        if y is None:
            y = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9] * 10, dtype='int32')
        if self.cfg.LAYER_COND and len(y.shape) == 1:
            y = np.eye(self.cfg.N_LABELS)[y]
        if not self.sampler_initialized:
            self._init_sampler()
        samples = self.session.run(self.sampler, feed_dict={self.z: z, self.y: y, self.t_train: False})
        samples = ((samples + 1.) * (255. / 2)).astype('int32')
        samples = samples.reshape((len(z), 3, self.cfg.OUTPUT_RES, self.cfg.OUTPUT_RES))
        return samples.transpose((0, 2, 3, 1))

    def z_sampler(self, size=None):
        if size is None:
            size = self.cfg.BATCH_SIZE
        return np.random.normal(size=(size, 128)).astype('float32')

    def sample_d(self, input, y):
        if self.cfg.LAYER_COND and len(y.shape) == 1:
            y = np.eye(self.cfg.N_LABELS)[y]
        if not self.sampler_initialized:
            self._init_sampler()
        return self.session.run(self.sampler_d, feed_dict={self.all_real_data_int: input,
                                                           self.y: y, self.t_train: False})

    def train(self):
        cfg = self.cfg
        t_True = tf.constant(True, dtype=tf.bool)
        t_False = tf.constant(False, dtype=tf.bool)
        DEVICES = ['/gpu:{}'.format(i) for i in xrange(cfg.N_GPUS)]
        if len(DEVICES) == 1:  # Hack because the code assumes 2 GPUs
            DEVICES = [DEVICES[0], DEVICES[0]]

        if cfg.LAYER_COND:
            labels_splits = tf.split(tf.one_hot(self.all_real_labels, depth=cfg.N_LABELS), len(DEVICES), axis=0)
        else:
            labels_splits = tf.split(self.all_real_labels, len(DEVICES), axis=0)

        fake_data_splits = []
        for i, device in enumerate(DEVICES):
            with tf.device(device):
                fake_data_splits.append(self.Generator(cfg, cfg.BATCH_SIZE / len(DEVICES), labels_splits[i],
                                                       is_training=self.t_train))

        all_real_data = tf.reshape(2 * ((tf.cast(self.all_real_data_int, tf.float32) / 256.) - .5),
                                   [cfg.BATCH_SIZE, cfg.OUTPUT_DIM])
        all_real_data += tf.random_uniform(shape=[cfg.BATCH_SIZE, cfg.OUTPUT_DIM], minval=0.,
                                           maxval=1. / 128)  # dequantize
        all_real_data_splits = tf.split(all_real_data, len(DEVICES), axis=0)

        DEVICES_B = DEVICES[:len(DEVICES) / 2]
        DEVICES_A = DEVICES[len(DEVICES) / 2:]

        disc_costs = []
        disc_acgan_costs = []
        disc_acgan_accs = []
        disc_acgan_fake_accs = []
        for i, device in enumerate(DEVICES_A):
            with tf.device(device):
                real_and_fake_data = tf.concat([
                    all_real_data_splits[i],
                    all_real_data_splits[len(DEVICES_A) + i],
                    fake_data_splits[i],
                    fake_data_splits[len(DEVICES_A) + i]
                ], axis=0)
                real_and_fake_labels = tf.concat([
                    labels_splits[i],
                    labels_splits[len(DEVICES_A) + i],
                    labels_splits[i],
                    labels_splits[len(DEVICES_A) + i]
                ], axis=0)
                disc_all, disc_all_acgan = self.Discriminator(cfg, real_and_fake_data, real_and_fake_labels)
                disc_real = disc_all[:cfg.BATCH_SIZE / len(DEVICES_A)]
                disc_fake = disc_all[cfg.BATCH_SIZE / len(DEVICES_A):]
                if cfg.MODE == 'wgan' or cfg.MODE == 'wgan-gp':
                    disc_costs.append(tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real))
                elif cfg.MODE == 'dcgan':
                    disc_costs.append(tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_real,
                                                                                             labels=tf.ones_like(
                                                                                                 disc_real))) / 2.)
                elif cfg.MODE == 'lsgan':
                    disc_costs.append(tf.reduce_mean((disc_real - 1) ** 2) / 2.)

                # ACGAN cost, if applicable
                if cfg.CONDITIONAL and cfg.ACGAN:
                    disc_acgan_costs.append(tf.reduce_mean(
                        tf.nn.sparse_softmax_cross_entropy_with_logits(
                            logits=disc_all_acgan[:cfg.BATCH_SIZE / len(DEVICES_A)],
                            labels=real_and_fake_labels[:cfg.BATCH_SIZE / len(DEVICES_A)])
                    ))
                    disc_acgan_accs.append(tf.reduce_mean(
                        tf.cast(
                            tf.equal(
                                tf.to_int32(tf.argmax(disc_all_acgan[:cfg.BATCH_SIZE / len(DEVICES_A)], dimension=1)),
                                real_and_fake_labels[:cfg.BATCH_SIZE / len(DEVICES_A)]
                            ),
                            tf.float32
                        )
                    ))
                    disc_acgan_fake_accs.append(tf.reduce_mean(
                        tf.cast(
                            tf.equal(
                                tf.to_int32(tf.argmax(disc_all_acgan[cfg.BATCH_SIZE / len(DEVICES_A):], dimension=1)),
                                real_and_fake_labels[cfg.BATCH_SIZE / len(DEVICES_A):]
                            ),
                            tf.float32
                        )
                    ))

        for i, device in enumerate(DEVICES_B):
            with tf.device(device):
                real_data = tf.concat([all_real_data_splits[i], all_real_data_splits[len(DEVICES_A) + i]], axis=0)
                fake_data = tf.concat([fake_data_splits[i], fake_data_splits[len(DEVICES_A) + i]], axis=0)
                labels = tf.concat([
                    labels_splits[i],
                    labels_splits[len(DEVICES_A) + i],
                ], axis=0)
                if cfg.MODE == 'wgan-gp':
                    alpha = tf.random_uniform(
                        shape=[cfg.BATCH_SIZE / len(DEVICES_A), 1],
                        minval=0.,
                        maxval=1.
                    )
                    differences = fake_data - real_data
                    interpolates = real_data + (alpha * differences)
                    gradients = tf.gradients(self.Discriminator(cfg, interpolates, labels)[0], [interpolates])[0]
                    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
                    gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
                    disc_costs.append(cfg.LAMBDA * gradient_penalty)
                elif cfg.MODE == 'dcgan':
                    disc_costs.append(tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake,
                                                                                             labels=tf.zeros_like(
                                                                                                 disc_fake))) / 2.)
                elif cfg.MODE == 'lsgan':
                    disc_costs.append(tf.reduce_mean((disc_fake - 0) ** 2) / 2.)

        disc_wgan = tf.add_n(disc_costs) / len(DEVICES_A)
        tf.summary.scalar('disc_cost', disc_wgan)
        if cfg.CONDITIONAL and cfg.ACGAN:
            disc_acgan = tf.add_n(disc_acgan_costs) / len(DEVICES_A)
            tf.summary.scalar('disc_acgan', disc_acgan)
            disc_acgan_acc = tf.add_n(disc_acgan_accs) / len(DEVICES_A)
            tf.summary.scalar('disc_acgan_acc', disc_acgan_acc)
            disc_acgan_fake_acc = tf.add_n(disc_acgan_fake_accs) / len(DEVICES_A)
            tf.summary.scalar('disc_acgan_fake_acc', disc_acgan_fake_acc)
            disc_cost = disc_wgan + (cfg.ACGAN_SCALE * disc_acgan)
        else:
            disc_cost = disc_wgan
        tf.summary.scalar('disc_cost', disc_cost)

        # ---- Generator costs ---- #
        gen_costs = []
        gen_acgan_costs = []
        for i, device in enumerate(DEVICES):
            with tf.device(device):
                n_samples = cfg.GEN_BS_MULTIPLE * cfg.BATCH_SIZE / len(DEVICES)
                fake_labels = tf.concat([labels_splits[(i + n) % len(DEVICES)] for n in range(cfg.GEN_BS_MULTIPLE)],
                                        axis=0)
                disc_fake, disc_fake_acgan = self.Discriminator(cfg, self.Generator(cfg, n_samples, fake_labels,
                                                                                    is_training=self.t_train),
                                                                fake_labels)
                if cfg.MODE == 'wgan' or cfg.MODE == 'wgan-gp':
                    gen_costs.append(-tf.reduce_mean(disc_fake))
                elif cfg.MODE == 'dcgan':
                    gen_costs.append(tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake,
                                                                                            labels=tf.ones_like(
                                                                                                disc_fake))))
                elif cfg.MODE == 'lsgan':
                    gen_costs.append(tf.reduce_mean((disc_fake - 1) ** 2))
                # ACGAN cost, if applicable
                if disc_fake_acgan is not None:
                    gen_acgan_costs.append(tf.reduce_mean(
                        tf.nn.sparse_softmax_cross_entropy_with_logits(logits=disc_fake_acgan, labels=fake_labels)
                    ))
        gen_cost = (tf.add_n(gen_costs) / len(DEVICES))
        if cfg.CONDITIONAL and cfg.ACGAN:
            gen_cost += (cfg.ACGAN_SCALE_G * (tf.add_n(gen_acgan_costs) / len(DEVICES)))
        gen_costs = gen_costs

        # ---- Optimizer functions ---- #

        if cfg.DECAY:
            decay = tf.maximum(0., 1. - (tf.cast(self._iteration, tf.float32) / cfg.ITERS))
        else:
            decay = 1.

        if cfg.MODE == 'wgan-gp':
            gen_opt = tf.train.AdamOptimizer(learning_rate=cfg.LR * decay, beta1=0., beta2=0.9)
            disc_opt = tf.train.AdamOptimizer(learning_rate=cfg.LR * decay, beta1=0., beta2=0.9)
        elif cfg.MODE == 'dcgan':
            gen_opt = tf.train.AdamOptimizer(learning_rate=cfg.LR * decay, beta1=0.5)
            disc_opt = tf.train.AdamOptimizer(learning_rate=cfg.LR * decay, beta1=0.5)
        elif cfg.MODE == 'wgan' or cfg.MODE == 'lsgan':
            gen_opt = tf.train.RMSPropOptimizer(learning_rate=cfg.LR * decay)
            disc_opt = tf.train.RMSPropOptimizer(learning_rate=cfg.LR * decay)
        gen_gv = gen_opt.compute_gradients(gen_cost, var_list=lib.params_with_name('Generator'))
        disc_gv = disc_opt.compute_gradients(disc_cost, var_list=lib.params_with_name('Discriminator.'))
        # add BN dependencies
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            gen_train_op = gen_opt.apply_gradients(gen_gv)
            disc_train_op = disc_opt.apply_gradients(disc_gv)

        if cfg.MODE == 'wgan':
            clip_ops = []
            for var in lib.params_with_name('Discriminator'):
                clip_bounds = [-.01, .01]
                clip_ops.append(tf.assign(var, tf.clip_by_value(var, clip_bounds[0], clip_bounds[1])))
            clip_disc_weights = tf.group(*clip_ops)

        # Function for generating samples
        # todo: check possibility to change this to none
        fixed_noise_samples = self.Generator(cfg, 100, self.fixed_labels, noise=self.fixed_noise,
                                             is_training=self.t_train)

        # Function for calculating inception score
        if self.label_probs is not None:
            elems = tf.convert_to_tensor(range(cfg.N_LABELS))
            samples = tf.multinomial(tf.log([self.label_probs]), 100)  # note log-prob
            fake_labels_100 = elems[tf.cast(samples[0], tf.int32)]
        else:
            fake_labels_100 = tf.cast(tf.random_uniform([100]) * cfg.N_LABELS, tf.int32)
        if cfg.LAYER_COND:
            fake_labels_100 = tf.cast(tf.one_hot(fake_labels_100, cfg.N_LABELS), tf.float32)

        samples_100 = self.Generator(cfg, 100, fake_labels_100, is_training=self.t_train)

        def get_inception_score(n):
            all_samples = []
            for i in xrange(n / 100):
                # todo
                all_samples.append(session.run(samples_100, feed_dict={self.t_train: True}))
            all_samples = np.concatenate(all_samples, axis=0)
            all_samples = ((all_samples + 1.) * (255.99 / 2)).astype('int32')
            all_samples = all_samples.reshape((-1, 3, cfg.OUTPUT_RES, cfg.OUTPUT_RES)).transpose(0, 2, 3, 1)
            return lib.inception_score.get_inception_score(list(all_samples))

        data_loader = self.get_data_loader()
        train_gen = data_loader.load_new(cfg)

        def inf_train_gen():
            while True:
                for images, _labels in train_gen():
                    yield images, _labels

        for name, grads_and_vars in [('G', gen_gv), ('D', disc_gv)]:
            print "{} Params:".format(name)
            total_param_count = 0
            for g, v in grads_and_vars:
                shape = v.get_shape()
                shape_str = ",".join([str(x) for x in v.get_shape()])

                param_count = 1
                for dim in shape:
                    param_count *= int(dim)
                total_param_count += param_count

                if g == None:
                    print "\t{} ({}) [no grad!]".format(v.name, shape_str)
                else:
                    print "\t{} ({})".format(v.name, shape_str)
            print "Total param count: {}".format(
                locale.format("%d", total_param_count, grouping=True)
            )
        run_number = len(next(os.walk(self.tb_dir))[1]) + 1
        summaries_merged = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(os.path.join(self.tb_dir, 'run_%i' % run_number), session.graph)
        # separate dev disc cost
        dev_cost_summary = tf.summary.scalar('dev_disc_cost', disc_cost)

        session.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver()
        self.restore_model()

        gen = inf_train_gen()
        sample_images, sample_labels = gen.next()
        if sample_labels is None:
            sample_labels = [0] * cfg.BATCH_SIZE

        # Save a batch of ground-truth samples
        _x_r = self.session.run(real_data, feed_dict={self.all_real_data_int: sample_images,
                                                      self.all_real_labels: sample_labels})
        _x_r = ((_x_r + 1.) * (255.99 / 2)).astype('int32')
        lib.save_images.save_images(_x_r.reshape((cfg.BATCH_SIZE, 3, cfg.OUTPUT_RES, cfg.OUTPUT_RES)),
                                    os.path.join(self.run_dir, 'samples_groundtruth.png'))

        # Function to generate samples
        def generate_image(log_dir, frame):
            samples = self.session.run(fixed_noise_samples, feed_dict={self.t_train: True})
            samples = ((samples + 1.) * (255. / 2)).astype('int32')
            lib.save_images.save_images(samples.reshape((100, 3, cfg.OUTPUT_RES, cfg.OUTPUT_RES)),
                                        os.path.join(log_dir, 'samples_{}.png'.format(frame)))

        if cfg.CONDITIONAL and cfg.ACGAN:
            _costs = {'cost': [], 'wgan': [], 'acgan': [], 'acgan_acc': [], 'acgan_fake_acc': []}
        else:
            _costs = {'cost': []}
        for iteration in xrange(self.current_iter, cfg.ITERS):
            start_time = time.time()

            if cfg.MODE == 'dcgan':
                gen_iters = cfg.N_GENERATOR
            else:
                gen_iters = 1
            if iteration > 0 and '_labels' in locals():
                for i in xrange(gen_iters):
                    _ = self.session.run([gen_train_op], feed_dict={self._iteration: iteration,
                                                                    self.all_real_labels: _labels,
                                                                    self.t_train: True})

            if (cfg.MODE == 'dcgan') or (cfg.MODE == 'lsgan'):
                disc_iters = 1
            else:
                disc_iters = cfg.N_CRITIC
            for i in xrange(disc_iters):
                _data, _labels = gen.next()
                if _labels is None:
                    _labels = [0] * cfg.BATCH_SIZE
                if cfg.CONDITIONAL and cfg.ACGAN:
                    _summary, _disc_cost, _disc_wgan, _disc_acgan, _disc_acgan_acc, _disc_acgan_fake_acc, _ = \
                        self.session.run([summaries_merged, disc_cost, disc_wgan, disc_acgan, disc_acgan_acc,
                                          disc_acgan_fake_acc, disc_train_op],
                                         feed_dict={self.all_real_data_int: _data, self.all_real_labels: _labels,
                                                    self._iteration: iteration, self.t_train: True})
                    _costs['cost'].append(_disc_cost)
                    _costs['wgan'].append(_disc_wgan)
                    _costs['acgan'].append(_disc_acgan)
                    _costs['acgan_acc'].append(_disc_acgan_acc)
                    _costs['acgan_fake_acc'].append(_disc_acgan_fake_acc)
                else:
                    _summary, _disc_cost, _ = self.session.run([summaries_merged, disc_cost, disc_train_op],
                                                               feed_dict={self.all_real_data_int: _data,
                                                                          self.all_real_labels: _labels,
                                                                          self._iteration: iteration,
                                                                          self.t_train: True})
                    _costs['cost'].append(_disc_cost)
                if cfg.MODE == 'wgan':
                    _ = self.session.run([clip_disc_weights])

            if iteration % cfg.SUMMARY_FREQUENCY == cfg.SUMMARY_FREQUENCY - 1:
                summary_writer.add_summary(_summary, iteration)

            if iteration % 100 == 99:
                _dev_cost_summary = self.session.run(dev_cost_summary, feed_dict={self.all_real_data_int: sample_images,
                                                                                  self.all_real_labels: sample_labels,
                                                                                  self.t_train: True})
                summary_writer.add_summary(_dev_cost_summary, iteration)
                generate_image(self.sample_dir, iteration)

            if (iteration < 500) or (iteration % 1000 == 999):
                # ideally we have the averages here
                prints = 'iter %i' % iteration
                for name, values in _costs.items():
                    prints += "\t{}\t{}".format(name, np.mean(values))
                    _costs[name] = []
                print(prints)

            if iteration % 2000 == 1999:
                if iteration + 1 // 2000 % ((cfg.ITERS / 2000) // cfg.KEEP_CHECKPOINTS) == 0:
                    # keep this checkpoint
                    self.saver.save(self.session, os.path.join(self.save_dir, 'model.ckpt'), global_step=iteration)
                else:
                    self.saver.save(self.session, os.path.join(self.save_dir, 'model.ckpt'))
                np.savetxt(os.path.join(self.save_dir, 'last_iteration'), [iteration])
                if cfg.bn_init:
                    cfg.bn_init = False
                    with open(os.path.join(self.run_dir, 'config.json'), 'w') as f:
                        json.dump(self.cfg.__dict__, f)
                print('Model saved.')

            if (cfg.INCEPTION_FREQUENCY != 0) and (iteration % cfg.INCEPTION_FREQUENCY == cfg.INCEPTION_FREQUENCY - 1):
                inception_score = get_inception_score(50000)
                print('INCEPTION SCORE:\t' + str(inception_score))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    config_dict = Config().__dict__
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--load_config', action='store', type=str, default=None)
    for key, value in config_dict.iteritems():
        if key != 'train':
            parser.add_argument('--' + key, action='store', type=type(value), default=None)
    args = parser.parse_args()
    arg_dict = {}
    for arg in vars(args):
        if arg != 'load_config':
            val = getattr(args, arg)
            if val is not None:
                arg_dict[arg] = val
    with tf.Session() as session:
        if args.load_config is not None:
            wgan = WGAN(session, load_config=args.load_config)
        else:
            wgan = WGAN(session, config_dict=arg_dict, train=args.train)
        if args.train:
            wgan.train()
