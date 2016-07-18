from functools import partial
from menpo.shape.pointcloud import PointCloud
from menpofit.builder import compute_reference_shape
from menpofit.builder import rescale_images_to_reference_shape
from menpofit.fitter import (noisy_shape_from_bounding_box,
                             align_shape_with_bounding_box)
from pathlib import Path

import joblib
import menpo.feature
import menpo.image
import menpo.io as mio
import numpy as np
import tensorflow as tf
import detect
import utils


def build_reference_shape(paths, diagonal=200):
    """Builds the reference shape.

    Args:
      paths: paths that contain the ground truth landmark files.
      diagonal: the diagonal of the reference shape in pixels.
    Returns:
      the reference shape.
    """
    landmarks = []
    for path in paths:
        path = Path(path).parent.as_posix()
        landmarks += [
            group.lms
            for group in mio.import_landmark_files(path, verbose=True)
            if group.lms.n_points == 68
        ]

    return compute_reference_shape(landmarks,
                                   diagonal=diagonal).points.astype(np.float32)


def grey_to_rgb(im):
    """Converts menpo Image to rgb if greyscale

    Args:
      im: menpo Image with 1 or 3 channels.
    Returns:
      Converted menpo `Image'.
    """
    assert im.n_channels in [1, 3]

    if im.n_channels == 3:
        return im

    im.pixels = np.vstack([im.pixels] * 3)
    return im


def align_reference_shape(reference_shape, bb):
    min_xy = tf.reduce_min(reference_shape, 0)
    max_xy = tf.reduce_max(reference_shape, 0)
    min_x, min_y = min_xy[0], min_xy[1]
    max_x, max_y = max_xy[0], max_xy[1]

    reference_shape_bb = tf.pack([[min_x, min_y], [max_x, min_y],
                                  [max_x, max_y], [min_x, max_y]])

    def norm(x):
        return tf.sqrt(tf.reduce_sum(tf.square(x - tf.reduce_mean(x, 0))))

    ratio = norm(bb) / norm(reference_shape_bb)
    return tf.add(
        (reference_shape - tf.reduce_mean(reference_shape_bb, 0)) * ratio,
        tf.reduce_mean(bb, 0),
        name='initial_shape')


def random_shape(gts, reference_shape, pca_model):
    """Generates a new shape estimate given the ground truth shape.

    Args:
      gts: a numpy array [num_landmarks, 2]
      reference_shape: a Tensor of dimensions [num_landmarks, 2]
      pca_model: A PCAModel that generates shapes.
    Returns:
      The aligned shape, as a Tensor [num_landmarks, 2].
    """

    def synthesize(lms):
        return detect.synthesize_detection(pca_model, menpo.shape.PointCloud(
            lms).bounding_box()).points.astype(np.float32)

    bb, = tf.py_func(synthesize, [gts], [tf.float32])
    shape = align_reference_shape(reference_shape, bb)
    shape.set_shape(reference_shape.get_shape())

    return shape


def get_noisy_init_from_bb(reference_shape, bb, noise_percentage=.02):
    """Roughly aligns a reference shape to a bounding box.

    This adds some uniform noise for translation and scale to the
    aligned shape.

    Args:
      reference_shape: a numpy array [num_landmarks, 2]
      bb: bounding box, a numpy array [4, ]
      noise_percentage: noise presentation to add.
    Returns:
      The aligned shape, as a numpy array [num_landmarks, 2]
    """
    bb = PointCloud(bb)
    reference_shape = PointCloud(reference_shape)

    bb = noisy_shape_from_bounding_box(
        reference_shape,
        bb,
        noise_percentage=[noise_percentage, 0, noise_percentage]).bounding_box(
        )

    return align_shape_with_bounding_box(reference_shape, bb).points


def load_images(paths, group=None, verbose=True):
    """Loads and rescales input images to the diagonal of the reference shape.

    Args:
      paths: a list of strings containing the data directories.
      reference_shape: a numpy array [num_landmarks, 2]
      group: landmark group containing the grounth truth landmarks.
      verbose: boolean, print debugging info.
    Returns:
      images: a list of numpy arrays containing images.
      shapes: a list of the ground truth landmarks.
      reference_shape: a numpy array [num_landmarks, 2].
      shape_gen: PCAModel, a shape generator.
    """
    pixels = []
    shapes = []
    bbs = []

    reference_shape = PointCloud(build_reference_shape(paths))

    for path in paths:
        if verbose:
            print('Importing data from {}'.format(path))

        for im in mio.import_images(path, verbose=verbose, as_generator=True):
            group = group or im.landmarks[group]._group_label

            bb_root = im.path.parent.relative_to(im.path.parent.parent.parent)
            if 'set' not in str(bb_root):
                bb_root = im.path.parent.relative_to(im.path.parent.parent)
            im.landmarks['bb'] = mio.import_landmark_file(str(Path(
                'bbs') / bb_root / (im.path.stem + '.pts')))
            im = im.crop_to_landmarks_proportion(0.3, group='bb')
            im = im.rescale_to_pointcloud(reference_shape, group=group)
            im = grey_to_rgb(im)
            pixels.append(im.pixels.transpose(1, 2, 0))
            shapes.append(im.landmarks[group].lms)
            bbs.append(im.landmarks['bb'].lms)

    pca_model = detect.create_generator(shapes, bbs)

    return pixels, shapes, reference_shape.points, pca_model


def load_image(path, reference_shape, is_training=False, group='PTS'):
    """Load an annotated image.

    In the directory of the provided image file, there
    should exist a landmark file (.pts) with the same
    basename as the image file.

    Args:
      path: a path containing an image file.
      reference_shape: a numpy array [num_landmarks, 2]
      is_training: whether in training mode or not.
      group: landmark group containing the grounth truth landmarks.
    Returns:
      pixels: a numpy array [width, height, 3].
      estimate: an initial estimate a numpy array [68, 2].
      gt_truth: the ground truth landmarks, a numpy array [68, 2].
    """
    im = mio.import_image(path)
    bb_root = im.path.parent.relative_to(im.path.parent.parent.parent)
    if 'set' not in str(bb_root):
        bb_root = im.path.parent.relative_to(im.path.parent.parent)

    im.landmarks['bb'] = mio.import_landmark_file(str(Path('bbs') / bb_root / (
        im.path.stem + '.pts')))

    im = im.crop_to_landmarks_proportion(0.3, group='bb')
    reference_shape = PointCloud(reference_shape)
    if np.random.rand() < .5:
        im = utils.mirror_image(im)

    bb = im.landmarks['bb'].lms.bounding_box()

    im.landmarks['__initial'] = align_shape_with_bounding_box(reference_shape,
                                                              bb)
    im = im.rescale_to_pointcloud(reference_shape, group='__initial')

    lms = im.landmarks[group].lms
    initial = im.landmarks['__initial'].lms

    # if the image is greyscale then convert to rgb.
    pixels = grey_to_rgb(im).pixels.transpose(1, 2, 0)

    gt_truth = lms.points.astype(np.float32)
    estimate = initial.points.astype(np.float32)
    return pixels.astype(np.float32).copy(), gt_truth, estimate


def distort_color(image, thread_id=0, scope=None):
    """Distort the color of the image.
    Each color distortion is non-commutative and thus ordering of the color ops
    matters. Ideally we would randomly permute the ordering of the color ops.
    Rather then adding that level of complication, we select a distinct ordering
    of color ops for each preprocessing thread.
    Args:
      image: Tensor containing single image.
      thread_id: preprocessing thread ID.
      scope: Optional scope for op_scope.
    Returns:
      color-distorted image
    """
    with tf.op_scope([image], scope, 'distort_color'):
        color_ordering = thread_id % 2

        if color_ordering == 0:
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.2)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        elif color_ordering == 1:
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.2)

        # The random_* ops do not necessarily clamp.
        image = tf.clip_by_value(image, 0.0, 1.0)
        return image


def batch_inputs(images,
                 shapes,
                 reference_shape,
                 batch_size=32,
                 is_training=False,
                 num_landmarks=68):
    """Reads the files off the disk and produces batches.

    Args:
      paths: a list of directories that contain training images and
        the corresponding landmark files.
      reference_shape: a numpy array [num_landmarks, 2]
      batch_size: the batch size.
      is_traininig: whether in training mode.
      num_landmarks: the number of landmarks in the training images.
    Returns:
      images: a tf tensor of shape [batch_size, width, height, 3].
      lms: a tf tensor of shape [batch_size, 68, 2].
      lms_init: a tf tensor of shape [batch_size, 68, 2].
    """

    files = tf.concat(0, [tf.matching_files(d) for d in paths])

    filename_queue = tf.train.string_input_producer(files,
                                                    shuffle=is_training,
                                                    capacity=1000)

    image, lms, lms_init = tf.py_func(
        partial(load_image, is_training=is_training),
        [filename_queue.dequeue(), reference_shape], # input arguments
        [tf.float32, tf.float32, tf.float32], # output types
        name='load_image'
    )

    # The image has always 3 channels.
    image.set_shape([None, None, 3])

    if is_training:
        image = distort_color(image)

    lms = tf.reshape(lms, [num_landmarks, 2])
    lms_init = tf.reshape(lms_init, [num_landmarks, 2])

    images, lms, inits = tf.train.batch([image, lms, lms_init],
                                        batch_size=batch_size,
                                        num_threads=4,
                                        capacity=1000,
                                        enqueue_many=False,
                                        dynamic_pad=True)

    return images, lms, inits
