import tensorflow as tf
import layers.utils as utils

class RotationEquivariantNonlinearity(tf.keras.layers.Layer):
    def __init__(self, nonlin = tf.nn.elu, **kwargs):
        self.nonlin = nonlin
        super().__init__(**kwargs)

    def build(self, input_shape):
        channels = input_shape[-2]
        self.representation_index = input_shape[-1]
        biases_initializer = None

        if self.representation_index != 1:
            self.bias = self.add_weight(
                name="bias",
                shape=[channels],
                dtype=tf.float32,
                initializer=biases_initializer)
    
    def call(self, input):
        if self.representation_index == 1:
            return self.nonlin(input)
        else:
            norm = utils.norm_with_epsilon(input, axis=-1)
            nonlin_out = self.nonlin(tf.nn.bias_add(norm, self.bias))
            factor = tf.divide(nonlin_out, norm)
            # Expand dims for representation index.
            return tf.multiply(input, tf.expand_dims(factor, axis=-1))

class Nonlinearity(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        self.nonlin = tf.nn.elu
        self.biases_initializer = None
        super(Nonlinearity, self).__init__(**kwargs)

    def build(self, input_shapes):
        n = 0
        self.sublayers = []
        for key in input_shapes:
            for i, shape in enumerate(input_shapes[key]):
                self.sublayers.append(RotationEquivariantNonlinearity())
                n += 1

    @tf.function
    def call(self, input_tensor_list):
        output_tensor_list = {0: [], 1: []}
        n = 0
        for key in input_tensor_list:
            for i, tensor in enumerate(input_tensor_list[key]):
                tensor_out = self.sublayers[n](tensor)
                n += 1
                m = 0 if tensor_out.get_shape().as_list()[-1] == 1 else 1
                output_tensor_list[m].append(tensor_out)
        return output_tensor_list
