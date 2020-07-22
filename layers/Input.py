import tensorflow as tf
import layers.utils as utils
import numpy as np

class Input(tf.keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        # radial basis functions
        rbf_low = 0.0
        rbf_high = 3.5
        rbf_count = 4
        self.rbf_spacing = (rbf_high - rbf_low) / rbf_count
        self.centers = tf.cast(np.linspace(rbf_low, rbf_high, rbf_count), tf.float32)            
        super(Input, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        self.batch_mode = len(input_shape) == 3

    @tf.function
    def call(self, inputs):
        def get_rij(row):
            # rij : [N, N, 3]
            return self.difference_matrix(row)

        def get_rbf(row):    
            # dij : [N, N]
            dij = self.distance_matrix(row)
            # rbf : [N, N, rbf_count]
            gamma = 1. / self.rbf_spacing
            rbf = tf.exp(-gamma * tf.square(tf.expand_dims(dij, axis=-1) - self.centers))
            return rbf

        def get_ones(row):
            return tf.ones(shape=(4, 1, 1))

        if self.batch_mode:
            return tf.map_fn(get_ones, inputs), tf.map_fn(get_rbf, inputs), tf.map_fn(get_rij, inputs)
        else:
            return get_ones(), get_rbf(inputs), get_rij(inputs)


    def difference_matrix(self, geometry):
        """
        Get relative vector matrix for array of shape [N, 3].

        Args:
            geometry: tf.Tensor with Cartesian coordinates and shape [N, 3]

        Returns:
            Relative vector matrix with shape [N, N, 3]
        """
        # [N, 1, 3]
        ri = tf.expand_dims(geometry, axis=1)
        # [1, N, 3]
        rj = tf.expand_dims(geometry, axis=0)
        # [N, N, 3]
        rij = ri - rj
        return rij


    def distance_matrix(self, geometry):
        """
        Get relative distance matrix for array of shape [N, 3].

        Args:
            geometry: tf.Tensor with Cartesian coordinates and shape [N, 3]

        Returns:
            Relative distance matrix with shape [N, N]
        """
        # [N, N, 3]
        rij = self.difference_matrix(geometry)
        # [N, N]
        dij = utils.norm_with_epsilon(rij, axis=-1)
        return dij
