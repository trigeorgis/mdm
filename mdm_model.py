from functools import partial

import slim
import tensorflow as tf
import data_provider
import utils

from slim import ops
from slim import scopes

def error_300w(images, dx, test_initial_shapes, threshold=0.05):
    final_lms = [sh.from_vector(sh.as_vector() + d) for sh, d in zip(test_initial_shapes, dx.sum(1))]
    true_lms = [im.landmarks['PTS'][None] for im in images]

    error_per_image = []

    for f, g in zip(final_lms,true_lms):
        interocular_dist = np.linalg.norm(g.points[36] - g.points[45])
        error_per_image.append(np.sum(np.linalg.norm(f.points - g.points, axis=-1)) / (f.points.shape[0] * interocular_dist))

    norms = np.array(error_per_image)
    return (norms < (threshold)).mean() * 100, np.mean(norms) * 100

def align_reference_shape(im, bb):
    import joblib
    reference_shape = joblib.load('reference_shape')
    mean_shape = tf.constant(reference_shape.points, tf.float32, name='reference_shape')
    mean_bb = tf.constant(reference_shape.bounding_box().points,  tf.float32, name='reference_bb')

    def norm(x):
        return tf.sqrt(tf.reduce_sum(tf.square(x - tf.reduce_mean(x, 0))))

    ratio = norm(bb) / norm(mean_bb)
    align_mean_shape = (mean_shape - tf.reduce_mean(mean_bb, 0)) * ratio + tf.reduce_mean(bb, 0)
    new_size = tf.to_int32(tf.to_float(tf.shape(im)[:2]) / ratio)
    return tf.image.resize_bilinear(tf.expand_dims(im, 0), new_size)[0, :, :, :], align_mean_shape / ratio, ratio

def normalized_rmse(pred, gt_truth):
    # TODO: assert shapes
    #       remove 68
    norm = tf.sqrt(tf.reduce_sum(((gt_truth[:, 36, :] - gt_truth[:, 45, :])**2), 1))

    return tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.square(pred - gt_truth), 2)), 1) / (norm * 68)

def conv_model(inputs, is_training=True, scope=''):

  # summaries or losses.
  net = {}

  with tf.op_scope([inputs], scope, 'mdm_conv'):
    with scopes.arg_scope([ops.conv2d, ops.fc], is_training=is_training):
      with scopes.arg_scope([ops.conv2d], activation=tf.nn.relu, padding='VALID'):
        net['conv_1'] = ops.conv2d(inputs, 32, [3, 3], scope='conv_1')
        net['pool_1'] = ops.max_pool(net['conv_1'], [2, 2])
        net['conv_2'] = ops.conv2d(net['pool_1'], 32, [3, 3], scope='conv_2')
        net['pool_2'] = ops.max_pool(net['conv_2'], [2, 2])

        crop_size = net['pool_2'].get_shape().as_list()[1:3]
        net['conv_2_cropped'] = utils.get_central_crop(net['conv_2'], box=crop_size)

        net['concat'] = tf.concat(3, [net['conv_2_cropped'], net['pool_2']])
  return net


def model(images, inits, num_iterations=4, num_patches=68, patch_shape=(24, 24), num_channels=3):
  batch_size = images.get_shape().as_list()[0]
  hidden_state = tf.zeros((batch_size, 512))
  dx = tf.zeros((batch_size, num_patches, 2))
  endpoints = {}
  dxs = []

  for step in range(num_iterations):
      with tf.device('/cpu:0'):
          patches = tf.image.extract_patches(images, tf.constant(patch_shape), inits+dx)

      endpoints['patches'] = patches
      with tf.variable_scope('convnet', reuse=step>0):
          net = conv_model(patches)
          ims = net['concat']

      ims = tf.reshape(ims, (batch_size, -1))

      with tf.variable_scope('rnn', reuse=step>0) as scope:
          hidden_state = slim.ops.fc(tf.concat(1, [ims, hidden_state]), 512, activation=tf.tanh)
          prediction = slim.ops.fc(hidden_state, num_patches * 2, scope='pred', activation=None)
          endpoints['prediction'] = prediction
      prediction = tf.reshape(prediction, (batch_size, num_patches, 2))
      dx += prediction
      dxs.append(dx)

  return inits + dx, dxs, endpoints
