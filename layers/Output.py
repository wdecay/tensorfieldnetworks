import tensorflow as tf

class Output(tf.keras.layers.Layer):
    def __init__(self, num_classes, **kwargs):
        self.num_classes = num_classes
        super(Output, self).__init__(**kwargs)

    def build(self, input_shape):
        tfn_output_shape = input_shape[0][0].as_list()

        self.fully_connected_layer = self.add_weight( 
                                            shape = [tfn_output_shape[-2], self.num_classes], 
                                            dtype=tf.float32)
        self.output_biases = self.add_weight(
            shape = [self.num_classes], dtype=tf.float32)

    @tf.function
    def call(self, input):
        tfn_scalars = input[0][0]
        tfn_output = tf.reduce_mean(tf.squeeze(tfn_scalars), axis=0)
        # output : [num_classes]
        output = tf.einsum('xy,x->y', self.fully_connected_layer, tfn_output) + self.output_biases
        # NOTE: this use of softmax is discouraged here,
        # but I was not able to figure out how to fit the model
        # with tf.nn.softmax_cross_entropy_with_logits
        return tf.nn.softmax(output)