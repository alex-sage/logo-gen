Logo Generation and Manipulation with Clustered Generative Adversarial Networks
===============================================================================

Code for reproducing experiments in ["Logo Generation and Manipulation with Clustered Generative Adversarial Networks"](https://arxiv.org/abs/1712.04407).
The models are mainly meant to work with data in HDF5 format (using [h5py](https://www.h5py.org/)) such as our [Large Logo Dataset](https://data.vision.ee.ethz.ch/sagea/lld/), but can easily be adapted to different input data formats (the WGAN models already accepts [CIFAR](https://www.cs.toronto.edu/~kriz/cifar.html) and [MNIST](http://yann.lecun.com/exdb/mnist/)).

This repository consists of two main parts:

### DCGAN

Our adaptation of DCGAN implementing layer conditioning for training with cluster labels. This is largely based on ["DCGAN in Tensorflow"](https://github.com/carpedm20/DCGAN-tensorflow)

### WGAN

A modified and extended version of the official TensorFlow code from ["Improved Training of Wasserstein GANs"](https://arxiv.org/abs/1704.00028).

### Detailed Instructions

For usage instructions and prerequisits, please refer to the individual readme's.

### vector.py

A note on the `vector.py` file: Both versions contain the code for the models themselves as well as a file called `vector.py`, which is meant to facilitate experimentation with a trained model, such as sampling it and performimg interpolations and vector arithmetic in latent space. At this time the two implementations differ slightly to integrate with the DCGAN and WGAN models. If there is sufficient interest, we might also make this a independent module that can be used with any GAN (or VaE) model that provides some specified interface. We believe this would save the research community a significant amount of work if its usage would spread. Please feel free to contact us should you be interesed in using and/or help us develop such a module.

## Pretrained Models

[DCGAN - LLD-icon with 100 AE clusters](https://data.vision.ee.ethz.ch/sagea/lld/data/model_DCGAN_LLD-icon_ae_100.zip)

[WGAN - LLD-icon with 128 RC clusters](https://data.vision.ee.ethz.ch/sagea/lld/data/model_WGAN_LLD-icon_rc_128.zip)

[WGAN - LLD-icon-sharp with 128 RC clusters](https://data.vision.ee.ethz.ch/sagea/lld/data/model_WGAN_LLD-icon-sharp_rc_128.zip)

[WGAN - LLD-icon-sharp with 16 RC clusters](https://data.vision.ee.ethz.ch/sagea/lld/data/model_WGAN_LLD-icon-sharp_rc_16.zip)

[WGAM - LLD-logo with 16 RC clusters](https://data.vision.ee.ethz.ch/sagea/lld/data/model_WGAN_LLD-logo_rc_64.zip)

## Large Logo Dataset
[Website of our LLD Dataset](https://data.vision.ee.ethz.ch/sagea/lld/)
