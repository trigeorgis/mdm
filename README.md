# MDM

A Tensorflow implementation of the Mnemonic Descent Method.

# Installation Instructions

## TensorFlow

Follow the installation instructions of Tensorflow at

https://www.tensorflow.org/versions/r0.9/get_started/os_setup.html#installing-from-sources

but use 

  git clone git@github.com:trigeorgis/tensorflow.git

as the TensorFlow repo. This is a fork of Tensorflow (#ff75787c) but it includes some
C++ specific kernels, such as for the extraction of patches.

## Menpo

We are an avid supporter of the Menpo project (http://www.menpo.org/) which we use
in various ways throughout the implementation.

Please look at the installation instructions at:

http://www.menpo.org/installation/

# Pretrained models

A pretrained model on 300W train set can be found at: https://www.doc.ic.ac.uk/~gt108/theano_mdm.pb

