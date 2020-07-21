import tensorflow as tf

class SelfInteractionSimple(tf.keras.layers.Layer):
  def __init__(self, output_dim, **kwargs):
    self.output_dim = output_dim
    super().__init__(**kwargs)

  def build(self, input_shape):
       #input_dim = inputs.get_shape().as_list()[-2]
    weights_initializer = tf.initializers.Orthogonal()
    self.batch_mode = len(input_shape) == 4
    #biases_initializer = tf.constant_initializer(0.)
    self.w = self.add_weight(
      shape=(self.output_dim, input_shape[-2]),
      dtype=tf.float32,
      initializer=weights_initializer,
      #regularizer=tf.keras.regularizers.l2(0.02),
      trainable=True)
    
  @tf.function
  def call(self, inputs, training=False):
    def process_row(row):
      return tf.transpose(
        tf.einsum('afi,gf->aig', row, self.w), perm=[0, 2, 1])
    if self.batch_mode:
      return tf.map_fn(process_row, inputs)
    else:
      return process_row(inputs)
    #return tf.transpose(
    #  tf.einsum('afi,gf->aig', inputs, self.w), perm=[0, 2, 1])


class SelfInteraction(tf.keras.layers.Layer):
  def __init__(self, output_dim, **kwargs):
    self.output_dim = output_dim
    super(SelfInteraction, self).__init__(**kwargs)

  def build(self, input_shape):
    n = 0
    self.sublayers = []
    for key in input_shape:
        for i, shape in enumerate(input_shape[key]):
          self.sublayers.append(SelfInteractionSimple(self.output_dim))
          n += 1


  @tf.function
  def call(self, input_tensor_list, training=False):
    output_tensor_list = {0: [], 1: []}
    n = 0
    for key in input_tensor_list:
        for i, tensor in enumerate(input_tensor_list[key]):
            tensor_out = self.sublayers[n](tensor)
            n += 1
            m = 0 if tensor_out.get_shape().as_list()[-1] == 1 else 1
            output_tensor_list[m].append(tensor_out)
    return output_tensor_list
