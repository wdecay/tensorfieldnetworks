import tensorflow as tf
import numpy as np
import scipy.linalg

FLOAT_TYPE = tf.float32
EPSILON = 1e-8

def norm_with_epsilon(input_tensor, axis=None, keep_dims=False):
    """
    Regularized norm

    Args:
        input_tensor: tf.Tensor

    Returns:
        tf.Tensor normed over axis
    """
    return tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(input_tensor), axis=axis, keepdims=keep_dims), EPSILON))
