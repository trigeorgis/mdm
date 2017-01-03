import tensorflow as tf


def normalized_rmse_2(pred, gt_truth):
    """Computes the interocular distance error for 68-points.

    This computes the Root-Mean-Squared-Error (RMSE) normalised
    by the interocular distance.

    Args:
      prediction: A tf `Tensor` of dimensions [n_samples, 68, 2].
      gt_truth: A tf `Tensor` of dimensions [n_samples, 68, 2].
    Returns:
      The normalised errors of each of the samples in a `Tensor`
      of dimensions [n_samples].
    """

    n_landmarks = 68
    left_eye_index = 36
    right_eye_index = 45

    assert pred.get_shape()[1] == gt_truth.get_shape()[1] == n_landmarks
    norm = tf.nn.l2_normalize(
        gt_truth[:, left_eye_index, :] - gt_truth[:, right_eye_index, :], 1)
    rmse = tf.nn.l2_normalize(pred - gt_truth, 1)

    return tf.reduce_sum(rmse, 1) / (norm * n_landmarks)

def normalized_rmse(pred, gt_truth):
    norm = tf.sqrt(1e-12 + tf.reduce_sum(((gt_truth[:, 36, :] - gt_truth[:, 45, :])**2), 1))

    return tf.reduce_sum(tf.sqrt(1e-12 + tf.reduce_sum(tf.square(pred - gt_truth), 2)), 1) / (norm * 68)