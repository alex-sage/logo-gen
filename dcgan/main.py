import os
import numpy as np

from model import DCGAN
from utils import pp, visualize, show_all_variables, interactive_interp

import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_integer("epoch", 25, "Epochs to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("input_height", None, "The size of image to use (will be center cropped). [108]")
flags.DEFINE_integer("input_width", None, "The size of image to use (will be center cropped). If None, same value as input_height [None]")
flags.DEFINE_integer("output_height", 64, "The size of the output images to produce [64]")
flags.DEFINE_integer("output_width", None, "The size of the output images to produce. If None, same value as output_height [None]")
flags.DEFINE_integer("c_dim", 3, "Dimension of image color. [3]")
flags.DEFINE_string("dataset", "LLD", "The name of dataset (used to name folders etc)")
flags.DEFINE_string("input_fname_pattern", "*data*.pkl", "Glob pattern of filename of input images [*]")
flags.DEFINE_string("input_type", 'pickle', 'Input type (file/pickle) [pickle]')
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", None, "Directory name to save the image samples [samples]")
flags.DEFINE_string("data_dir", "/home/sagea/scratch/data/icons", "Directory path with favicon data")
flags.DEFINE_boolean("is_train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("is_crop", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", True, "True for visualizing, False for nothing [True]")
flags.DEFINE_boolean("interpolate", False, "True for interactively interpolating between two samples [False]")
flags.DEFINE_integer("num_g_updates", 2, "Number of generator updates before a discriminator update [2]")
flags.DEFINE_integer("z_dim", 100, "Dimension of Z-vector [100]")
flags.DEFINE_integer("gf_dim", 64, "Dimension of gen filters in first conv layer. [64]")
flags.DEFINE_integer("gf_size", 5, "Size of gen filters. [5]")
flags.DEFINE_integer("df_dim", 64, "Dimension of discrim filters in first conv layer. [64]")
flags.DEFINE_integer("df_size", 5, "Size of discrim filters. [5]")
flags.DEFINE_integer("gfc_dim", 1024, "Dimension of gen units for for fully connected layer. [1024]")
flags.DEFINE_integer("dfc_dim", 1024, "Dimension of discrim units for fully connected layer. [1024]")
flags.DEFINE_string("labels", None, "Specifies the HDF5 path for the labels, if any")
flags.DEFINE_string("sampling", 'uniform', "Sampling method (uniform/normal/t-normal/sphere) ['uniform']")
flags.DEFINE_string("sigma", 0.5, "Sigma used for truncated normal (t-normal) distribution [0.5]")
flags.DEFINE_float("gauss_sigma", 0, "Sigma for gaussian blurring [0]")
flags.DEFINE_float("blur_input", None, "Sigma for blurring input images [None]")
flags.DEFINE_integer("gauss_trunc", 2, "Defines generated kernel size for gaussian filtering as 2*trunc+1 [2]")
flags.DEFINE_bool("blur_fade", False, "If true, the gaussian blur on images slowly fades during training [False]")
flags.DEFINE_bool("ipython", False, "Switches to interactive command line")
flags.DEFINE_integer("y_dim", None, "number of labels [y_dim]")
FLAGS = flags.FLAGS


def main(_):
    pp.pprint(flags.FLAGS.__flags)

    # only width needed for square images
    if FLAGS.input_width is None:
        FLAGS.input_width = FLAGS.input_height
    if FLAGS.output_width is None:
        FLAGS.output_width = FLAGS.output_height
    # if no sample dir, set to subdir with dataset name
    if FLAGS.sample_dir is None:
        FLAGS.sample_dir = os.path.join('samples', FLAGS.dataset)
    # make sure set paths exist
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)
    # during training it makes no sense to have interpolation active
    if FLAGS.is_train:
        FLAGS.interpolate = False
    if FLAGS.interpolate:
        FLAGS.visualize = False
    if FLAGS.ipython:
        FLAGS.visualize = False

    # allows specifying additional options
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True

    # initialise session with chosen flags
    with tf.Session(config=run_config) as sess:
        dcgan = DCGAN(
                sess,
                input_width=FLAGS.input_width,
                input_height=FLAGS.input_height,
                output_width=FLAGS.output_width,
                output_height=FLAGS.output_height,
                batch_size=FLAGS.batch_size,
                sample_num=FLAGS.batch_size,
                c_dim=FLAGS.c_dim,
                dataset_name=FLAGS.dataset,
                input_fname_pattern=FLAGS.input_fname_pattern,
                is_crop=FLAGS.is_crop,
                checkpoint_dir=FLAGS.checkpoint_dir,
                sample_dir=FLAGS.sample_dir,
                num_g_updates=FLAGS.num_g_updates,
                z_dim=FLAGS.z_dim,
                gf_dim=FLAGS.gf_dim,
                gf_size=FLAGS.gf_size,
                df_dim=FLAGS.df_dim,
                df_size=FLAGS.df_size,
                gfc_dim=FLAGS.gfc_dim,
                dfc_dim=FLAGS.dfc_dim,
                data_dir=FLAGS.data_dir,
                is_train=FLAGS.is_train,
                label_path=FLAGS.labels,
                gauss_sigma=FLAGS.gauss_sigma,
                gauss_trunc=FLAGS.gauss_trunc,
                blur_fade=FLAGS.blur_fade,
                y_dim=FLAGS.y_dim)
        show_all_variables()
        if FLAGS.is_train:
            dcgan.train(FLAGS)
        else:
            if not dcgan.load(FLAGS.checkpoint_dir):
                raise Exception("[!] Train a model first, then run test mode")
        # visualisation
        if FLAGS.visualize:
            option = 2
            visualize(sess, dcgan, FLAGS, option)
        if FLAGS.interpolate:
            interactive_interp(sess, dcgan, FLAGS, sampling='uniform')
        if FLAGS.ipython:
            from vector import Vector
            from IPython import embed
            vec = Vector(sess, dcgan, FLAGS)
            embed()


if __name__ == '__main__':
    tf.app.run()
