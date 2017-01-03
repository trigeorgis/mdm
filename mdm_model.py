import tensorflow as tf
import data_provider
import utils
import numpy as np
from functools import partial
slim = tf.contrib.slim

_extract_patches_module = tf.load_op_library('/homes/gt108/Projects/tf_extract_patches/extract_patches.so')

def convolutional_model_mini(inputs):   
  conv_settings = dict(
    padding='VALID', num_outputs=32, kernel_size=3,
    weights_initializer=slim.initializers.xavier_initializer_conv2d(False),
    weights_regularizer=slim.l2_regularizer(5e-5)
  )

  with tf.variable_scope('convnet'):
    with slim.arg_scope([slim.conv2d], **conv_settings):
      with slim.arg_scope([slim.max_pool2d], kernel_size=2):
        net = slim.conv2d(inputs, scope='conv_1')
        net = slim.max_pool2d(net)
        conv_2 = slim.conv2d(net, scope='conv_2')
        pool_2 = slim.max_pool2d(conv_2)


        crop_size = pool_2.get_shape().as_list()[1:3]
        conv_2_cropped = utils.get_central_crop(conv_2, box=crop_size)

        net = tf.concat(3, [pool_2, conv_2_cropped])

  return net

def convolutional_model_mini_2(inputs):
    
  conv_settings = dict(
    padding='SAME', num_outputs=32, kernel_size=3,
    weights_initializer=slim.initializers.xavier_initializer_conv2d(False),
    weights_regularizer=slim.l2_regularizer(5e-5)
  )

  with tf.variable_scope('convnet'):
    with slim.arg_scope([slim.conv2d], **conv_settings):
      with slim.arg_scope([slim.max_pool2d], kernel_size=2):
        conv_1 = slim.conv2d(inputs, scope='conv_1')
        net = slim.max_pool2d(conv_1)
        net = slim.conv2d(net, scope='conv_2')
        pool_2 = slim.max_pool2d(net)


        crop_size = pool_2.get_shape().as_list()[1:3]
        conv_1_cropped = utils.get_central_crop(conv_1, box=crop_size)

        net = tf.concat(3, [pool_2, conv_1_cropped])
        net = slim.conv2d(net, scope='conv_3')
  return net

def convolutional_model_mini_3(inputs):
    
  conv_settings = dict(
    padding='SAME', num_outputs=32, kernel_size=3,
    weights_initializer=slim.initializers.xavier_initializer_conv2d(False),
    weights_regularizer=slim.l2_regularizer(5e-5)
  )

  with tf.variable_scope('convnet'):
    with slim.arg_scope([slim.conv2d], **conv_settings):
      with slim.arg_scope([slim.max_pool2d], kernel_size=2):
        net = slim.conv2d(inputs, scope='conv_1')
        net = slim.max_pool2d(net)
        net = slim.conv2d(net, scope='conv_2')
        net = slim.max_pool2d(net)
        net = slim.conv2d(net, scope='conv_3')
  return net

def convolutional_model(inputs):
  with tf.variable_scope('convnet'):
    with slim.arg_scope([slim.conv2d], padding='SAME', num_outputs=48, kernel_size=3, normalizer_fn=slim.batch_norm):
      with slim.arg_scope([slim.max_pool2d], kernel_size=2):
        net = slim.conv2d(inputs, scope='conv_1')
        net = slim.max_pool2d(net)
        net = slim.conv2d(net, scope='conv_2')
        net = slim.max_pool2d(net)
        net = slim.conv2d(net, scope='conv_3')
        net = slim.conv2d(net, scope='conv_3_2')
        net = slim.max_pool2d(net)
        net = slim.conv2d(net, scope='conv_4')
        net = slim.conv2d(net, scope='conv_4_2')
        net = slim.max_pool2d(net)

  return net

def build_sampling_grid(patch_shape):
    patch_shape = np.array(patch_shape)
    patch_half_shape = np.require(np.round(patch_shape / 2), dtype=int)
    start = -patch_half_shape
    end = patch_half_shape
    sampling_grid = np.mgrid[start[0]:end[0], start[1]:end[1]]
    return sampling_grid.swapaxes(0, 2).swapaxes(0, 1)

default_sampling_grid = build_sampling_grid((30, 30))

def extract_patches_image(image, centres, sampling_grid=default_sampling_grid):
    """ Extracts patches from an image.
    Args:
        pixels: a `Tensor` of dimensions [batch_size, height, width, channels]
        centres: a `Tensor` of dimensions [batch_size, num_patches, 2]
        sampling_grid: `ndarray` (patch_width, patch_height, 2)
    Returns:
        a `Tensor` [num_patches, height, width, channels]
    """

    max_y = tf.shape(image)[0]
    max_x = tf.shape(image)[1]

    patch_grid = tf.to_int32(sampling_grid[None, :, :, :] + centres[:,  None, None, :])
    Y = tf.clip_by_value(patch_grid[:, :, :, 0], 0, max_y - 1)
    X = tf.clip_by_value(patch_grid[:, :, :, 1], 0, max_x - 1)

    return tf.gather_nd(image, tf.transpose(tf.pack([Y, X]), (2, 3, 1, 0)))

def extract_patches(images, centres, sampling_grid=default_sampling_grid):
    batch_size = images.get_shape().as_list()[0]
    patches = tf.pack([extract_patches_image(images[i], centres[i], sampling_grid=sampling_grid)
                       for i in range(batch_size)])
    return tf.transpose(patches, [0, 3, 1, 2, 4])

def model(images, initial_shapes, num_iterations=3, num_patches=68, patch_shape=(30, 30), hidden_size=256, num_channels=3):
  sampling_grid = build_sampling_grid(patch_shape)
  batch_size = images.get_shape().as_list()[0]
  hidden_state = tf.zeros((batch_size, hidden_size))
  deltas = tf.zeros((batch_size, num_patches, 2))
  predictions = []

  for step in range(num_iterations):
      with tf.device('/cpu:0'):
          # patches = _extract_patches_module.extract_patches(images, tf.constant(patch_shape), initial_shapes + deltas)
          patches = extract_patches(images, initial_shapes + deltas, sampling_grid=sampling_grid)
          # TODO: Implement the gradient.
          patches = tf.stop_gradient(patches)

      # patches = tf.reshape(patches, (batch_size * num_patches, patch_shape[0], patch_shape[1], num_channels))
      patches = tf.reshape(patches, (batch_size , num_patches * patch_shape[0], patch_shape[1], num_channels))

      
      with tf.variable_scope('convnet', reuse=step > 0):
          features = convolutional_model_mini_3(patches)

      features = tf.reshape(features, (batch_size, -1))

      with tf.variable_scope('rnn', reuse=step>0) as scope:
          with slim.arg_scope([slim.fully_connected], weights_regularizer=slim.l2_regularizer(5e-5)):
              hidden_state = slim.fully_connected(tf.concat(1, [features, hidden_state]), hidden_size, activation_fn=tf.tanh)
          prediction = slim.linear(hidden_state, num_patches * 2, scope='pred')

      prediction = tf.reshape(prediction, (batch_size, num_patches, 2))
      deltas += prediction
      predictions.append(initial_shapes + deltas)

  return predictions
