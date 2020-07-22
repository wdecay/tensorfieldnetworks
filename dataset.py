import numpy as np
import tensorflow as tf
import numpy as np
import scipy
from layers.utils import EPSILON

tetris = [[(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 1, 0)],  # chiral_shape_1
          [(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, -1, 0)], # chiral_shape_2
          [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)],  # square
          [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3)],  # line
          [(0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0)],  # corner
          [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 0)],  # T
          [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 1)],  # zigzag
          [(0, 0, 0), (1, 0, 0), (1, 1, 0), (2, 1, 0)]]  # L


def random_rotation_matrix(numpy_random_state):
    """
    Generates a random 3D rotation matrix from axis and angle.

    Args:
        numpy_random_state: numpy random state object

    Returns:
        Random rotation matrix.
    """
    rng = numpy_random_state
    axis = rng.randn(3)
    axis /= np.linalg.norm(axis) + EPSILON
    theta = 2 * np.pi * rng.uniform(0.0, 1.0)
    return rotation_matrix(axis, theta)


def rotation_matrix(axis, theta):
    return scipy.linalg.expm(np.cross(np.eye(3), axis * theta))

def get_rotation_augmentor(rng):
    def augment(shape, label):
        shape = tf.matmul(shape,
                          tf.convert_to_tensor(
                              random_rotation_matrix(rng),
                              dtype=tf.float32))
        return shape, label
    return augment

def get_translation_augmentor(rng):
    def augment(shape, label):
        shape = shape + np.expand_dims(rng.uniform(
            low=-3., high=3., size=(3)), axis=0)
        return shape, label
    return augment    

def get_dataset():
    num_classes = len(tetris)
    x = np.array([np.array(points_, dtype='float32') for points_ in tetris])
    y = np.array([i for i in range(num_classes)])
    return tf.data.Dataset.from_tensor_slices((x, y)), len(tetris)

