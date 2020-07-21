import tensorflow as tf

class Concatenation(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Concatenation, self).__init__(**kwargs)

    @tf.function
    def call(self, input_tensor_list):
        output_tensor_list = {0: [], 1: []}
        for key in input_tensor_list:
            output_tensor_list[key].append(tf.concat(input_tensor_list[key], axis=-2))
        return output_tensor_list
