from datetime import datetime
import data_provider
import joblib
import mdm_model
import numpy as np
import os.path
import slim
import tensorflow as tf
import time
import utils
import menpo
import menpo.io as mio

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float('initial_learning_rate', 0.001,
                          """Initial learning rate.""")
tf.app.flags.DEFINE_float('num_epochs_per_decay', 5.0,
                          """Epochs after which learning rate decays.""")
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.97,
                          """Learning rate decay factor.""")
tf.app.flags.DEFINE_integer('batch_size', 40, """The batch size to use.""")
tf.app.flags.DEFINE_integer('num_preprocess_threads', 4,
                            """How many preprocess threads to use.""")
tf.app.flags.DEFINE_string('train_dir', 'ckpt/train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('pretrained_model_checkpoint_path', '',
                           """If specified, restore this pretrained model """
                           """before beginning any training.""")
tf.app.flags.DEFINE_integer('max_steps', 50000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_string('train_device', '/gpu:0', """Device to train with.""")
tf.app.flags.DEFINE_string('datasets', ':'.join(
    ('databases/lfpw/trainset/*.png', 'databases/afw/*.jpg',
     'databases/helen/trainset/*.jpg')),
                           """Directory where to write event logs """
                           """and checkpoint.""")
# The decay to use for the moving average.
MOVING_AVERAGE_DECAY = 0.9999


def train(scope=''):
    """Train on dataset for a number of steps."""
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        # Create a variable to count the number of train() calls. This equals the
        # number of batches processed * FLAGS.num_gpus.
        global_step = tf.get_variable('global_step', [],
                                      initializer=tf.constant_initializer(0),
                                      trainable=False)

        train_dirs = FLAGS.datasets.split(':')

        # Calculate the learning rate schedule.
        num_batches_per_epoch = 100
        num_epochs_per_decay = 5
        decay_steps = int(num_batches_per_epoch * num_epochs_per_decay)

        # Decay the learning rate exponentially based on the number of steps.
        lr = tf.train.exponential_decay(FLAGS.initial_learning_rate,
                                        global_step,
                                        decay_steps,
                                        FLAGS.learning_rate_decay_factor,
                                        staircase=True)

        # Create an optimizer that performs gradient descent.
        opt = tf.train.AdamOptimizer(lr)

        # Override the number of preprocessing threads to account for the increased
        # number of GPU towers.
        num_preprocess_threads = FLAGS.num_preprocess_threads

        _images, _shapes, _reference_shape, pca_model = \
            data_provider.load_images(train_dirs)

        reference_shape = tf.constant(_reference_shape,
                                      dtype=tf.float32,
                                      name='reference_shape')

        image_shape = _images[0].shape
        lms_shape = _shapes[0].points.shape

        def get_random_sample(rotation_stddev=10):
            idx = np.random.randint(low=0, high=len(_images))
            im = menpo.image.Image(_images[idx].transpose(2, 0, 1))
            lms = _shapes[idx]
            im.landmarks['PTS'] = lms

            if np.random.rand() < .5:
                im = utils.mirror_image(im)

            theta = np.random.normal(scale=rotation_stddev)

            rot = menpo.transform.rotate_ccw_about_centre(lms, theta)
            im = im.warp_to_shape(im.shape, rot)

            pixels = im.pixels.transpose(1, 2, 0).astype('float32')
            shape = im.landmarks['PTS'].lms.points.astype('float32')
            return pixels, shape

        image, shape = tf.py_func(get_random_sample, [],
                                  [tf.float32, tf.float32])

        initial_shape = data_provider.random_shape(shape, reference_shape,
                                                   pca_model)
        image.set_shape(image_shape)
        shape.set_shape(lms_shape)
        initial_shape.set_shape(lms_shape)

        image = data_provider.distort_color(image)

        images, lms, inits = tf.train.batch([image, shape, initial_shape],
                                            FLAGS.batch_size,
                                            dynamic_pad=False,
                                            capacity=5000,
                                            enqueue_many=False,
                                            num_threads=num_preprocess_threads,
                                            name='batch')
        print('Defining model...')
        with tf.device(FLAGS.train_device):
            # Retain the summaries from the final tower.
            summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
            predictions, dxs, _ = mdm_model.model(images, inits)

            total_loss = 0

            for i, dx in enumerate(dxs):
                norm_error = mdm_model.normalized_rmse(dx + inits, lms)
                tf.histogram_summary('errors', norm_error)
                loss = tf.reduce_mean(norm_error)
                total_loss += loss
                summaries.append(tf.scalar_summary('losses/step_{}'.format(i),
                                                   loss))

            # Calculate the gradients for the batch of data
            grads = opt.compute_gradients(total_loss)

        summaries.append(tf.scalar_summary('losses/total', total_loss))
        pred_images, = tf.py_func(utils.batch_draw_landmarks,
                                  [images, predictions], [tf.float32])
        gt_images, = tf.py_func(utils.batch_draw_landmarks, [images, lms],
                                [tf.float32])

        summary = tf.image_summary('images',
                                   tf.concat(2, [gt_images, pred_images]),
                                   max_images=5)
        summaries.append(tf.histogram_summary('dx', predictions - inits))

        summaries.append(summary)

        batchnorm_updates = tf.get_collection(slim.ops.UPDATE_OPS_COLLECTION,
                                              scope)

        # Add a summary to track the learning rate.
        summaries.append(tf.scalar_summary('learning_rate', lr))

        # Add histograms for gradients.
        for grad, var in grads:
            if grad is not None:
                summaries.append(tf.histogram_summary(var.op.name +
                                                      '/gradients', grad))

        # Apply the gradients to adjust the shared variables.
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

        # Add histograms for trainable variables.
        for var in tf.trainable_variables():
            summaries.append(tf.histogram_summary(var.op.name, var))

        # Track the moving averages of all trainable variables.
        # Note that we maintain a "double-average" of the BatchNormalization
        # global statistics. This is more complicated then need be but we employ
        # this for backward-compatibility with our previous models.
        variable_averages = tf.train.ExponentialMovingAverage(
            MOVING_AVERAGE_DECAY, global_step)

        # Another possibility is to use tf.slim.get_variables().
        variables_to_average = (
            tf.trainable_variables() + tf.moving_average_variables())
        variables_averages_op = variable_averages.apply(variables_to_average)

        # Group all updates to into a single train op.
        # NOTE: Currently we are not using batchnorm in MDM.
        batchnorm_updates_op = tf.group(*batchnorm_updates)
        train_op = tf.group(apply_gradient_op, variables_averages_op,
                            batchnorm_updates_op)

        # Create a saver.
        saver = tf.train.Saver(tf.all_variables())

        # Build the summary operation from the last tower summaries.
        summary_op = tf.merge_summary(summaries)
        # Start running operations on the Graph. allow_soft_placement must be
        # set to True to build towers on GPU, as some of the ops do not have GPU
        # implementations.
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        # Build an initialization operation to run below.
        init = tf.initialize_all_variables()
        print('Initializing variables...')
        sess.run(init)
        print('Initialized variables.')

        if FLAGS.pretrained_model_checkpoint_path:
            assert tf.gfile.Exists(FLAGS.pretrained_model_checkpoint_path)
            variables_to_restore = tf.get_collection(
                slim.variables.VARIABLES_TO_RESTORE)
            restorer = tf.train.Saver(variables_to_restore)
            restorer.restore(sess, FLAGS.pretrained_model_checkpoint_path)
            print('%s: Pre-trained model restored from %s' %
                  (datetime.now(), FLAGS.pretrained_model_checkpoint_path))

        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess)

        summary_writer = tf.train.SummaryWriter(FLAGS.train_dir)

        print('Starting training...')
        for step in xrange(FLAGS.max_steps):
            start_time = time.time()
            _, loss_value = sess.run([train_op, total_loss])
            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step % 10 == 0:
                examples_per_sec = FLAGS.batch_size / float(duration)
                format_str = (
                    '%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                    'sec/batch)')
                print(format_str % (datetime.now(), step, loss_value,
                                    examples_per_sec, duration))

            if step % 10 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)

            # Save the model checkpoint periodically.
            if step % 50 == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)


if __name__ == '__main__':
    train()
