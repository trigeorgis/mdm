from datetime import datetime
import data_provider
import mdm_model
import tensorflow as tf
import utils
import losses
import menpo.io as mio

from pathlib import Path
from tensorflow.python.platform import tf_logging as logging

slim = tf.contrib.slim

use_pertubed = True
__training_datasets = ('lfpw_trainset', 'lfpw_testset', 'helen_trainset',
                       'helen_testset', 'ibug', 'afw')
__datasets_dir = Path('/vol/atlas/databases/tf_records')
__training_paths = [str(__datasets_dir / (('pertubed_' if use_pertubed else '') + x + '.tfrecords')) for x in __training_datasets]

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float('initial_learning_rate', 1e-3, 'Initial learning rate.')
tf.app.flags.DEFINE_float('decay_steps', 15000, 'Number of steps to decay learning rate.')
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.1, 'Learning rate decay factor.')
tf.app.flags.DEFINE_integer('batch_size', 40, '''The batch size to use.''')
tf.app.flags.DEFINE_integer('num_preprocess_threads', 4, 'How many preprocess threads to use.')
tf.app.flags.DEFINE_string('train_dir', 'ckpt/train',
                           '''Directory where to write event logs '''
                           '''and checkpoint.''')
tf.app.flags.DEFINE_string('pretrained_model_checkpoint_path', '',
                           '''If specified, restore this pretrained model '''
                           '''before beginning any training.''')
tf.app.flags.DEFINE_integer('max_steps', 100000, 'Number of batches to run.')
tf.app.flags.DEFINE_string('train_device', '/gpu:0', 'Device to train with.')
tf.app.flags.DEFINE_string('datasets', ':'.join(__training_paths), 
                           '''The datasets to use (tfrecords).''')
tf.app.flags.DEFINE_integer('patch_size', 30, 'The extracted patch size')

# The decay to use for the moving average.
MOVING_AVERAGE_DECAY = 0.9999

def draw_landmarks(images, landmarks):
    return tf.py_func(utils.batch_draw_landmarks,
               [images, landmarks], [tf.float32])[0]

def train():
    g = tf.Graph()

    with g.as_default():
        # Load dataset.
        datasets = FLAGS.datasets.split(':')
        images, gt_shapes, initial_shapes  = data_provider.batch_inputs(
            datasets, batch_size=FLAGS.batch_size, is_training=True)

        # Define model graph.
        patch_shape = (FLAGS.patch_size, FLAGS.patch_size)
        with slim.arg_scope(
            [slim.batch_norm, slim.layers.dropout], is_training=True):
            predictions = mdm_model.model(images, initial_shapes, patch_shape=patch_shape)

        for i, prediction in enumerate(predictions):
            norm_error = losses.normalized_rmse(prediction, gt_shapes)
            mse_error = tf.reduce_mean((prediction - initial_shapes)**2, 0)
            tf.histogram_summary('errors/step_{}'.format(i), mse_error)

            norm_error = slim.losses.compute_weighted_loss(norm_error)

            tf.scalar_summary('losses/step_{}'.format(i), tf.reduce_mean(norm_error))

        total_loss = slim.losses.get_total_loss()
        tf.scalar_summary('losses/total loss', total_loss)

        gt_images = draw_landmarks(images, gt_shapes)
        tf.image_summary('gt_images', gt_images)
        init_images = draw_landmarks(images, initial_shapes)
        pred_images = [init_images] + [draw_landmarks(images, x) for x in predictions]
        tf.image_summary('predictions', tf.concat(2, pred_images))
       
        # Calculate the learning rate schedule.
        decay_steps = 15000

        global_step = slim.get_global_step() or slim.create_global_step()

        # Decay the learning rate exponentially based on the number of steps.
        lr = tf.train.exponential_decay(FLAGS.initial_learning_rate,
                                        global_step,
                                        decay_steps,
                                        FLAGS.learning_rate_decay_factor,
                                        staircase=True)

        optimizer = tf.train.AdamOptimizer(lr)

    with tf.Session(graph=g) as sess:
        init_fn = None

        if FLAGS.pretrained_model_checkpoint_path:
            print('Loading whole model...')
            variables_to_restore = slim.get_model_variables()
            init_fn = slim.assign_from_checkpoint_fn(
                FLAGS.pretrained_model_checkpoint_path,
                variables_to_restore,
                ignore_missing_vars=True)

        train_op = slim.learning.create_train_op(
            total_loss, optimizer, summarize_gradients=True)

        logging.set_verbosity(1)
        slim.learning.train(
            train_op,
            FLAGS.train_dir,
            save_summaries_secs=60,
            init_fn=init_fn,
            save_interval_secs=600)


if __name__ == '__main__':
    train()