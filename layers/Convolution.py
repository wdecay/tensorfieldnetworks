import tensorflow as tf
import numpy as np
import layers.utils as utils
from layers.utils import FLOAT_TYPE, EPSILON

class Filter(tf.keras.layers.Layer):
  def __init__(self,  li, lf = None, **kwargs):
    super().__init__(**kwargs)
    self.li = li
    self.lf = lf
    self.nonlin = tf.nn.relu
    self.eijk = self.get_eijk()

  def build(self, input_shapes):
    layer_input_shape, rbf_shape, rij_shape = input_shapes
    weights_initializer = tf.keras.initializers.glorot_uniform()
    biases_initializer = tf.constant_initializer(0.)
    self.input_dim = layer_input_shape[-1]
    self.output_dim = layer_input_shape[-2]

    self.batch_mode = len(layer_input_shape) == 4
    hidden_dim = rbf_shape[-1]
    
    self.w1 = self.add_weight(shape=[hidden_dim, rbf_shape[-1]], dtype=FLOAT_TYPE,
                              initializer=weights_initializer, name="w1")
    self.w2 = self.add_weight(shape=[self.output_dim, hidden_dim], dtype=FLOAT_TYPE,
                              initializer=weights_initializer, name="w2")
    self.b1 = self.add_weight(shape=[hidden_dim], dtype=FLOAT_TYPE,
                              initializer=biases_initializer, name="b1")
    self.b2 = self.add_weight(shape=[self.output_dim], dtype=FLOAT_TYPE,
                              initializer=biases_initializer, name="b2")

  def call(self, inputs):
    def process_row(row):
      layer_input, rbf, rij = row
      if self.li == 0:
        return  self.filter_0(layer_input, rbf), 0, 0
      elif self.li == 1 and self.lf == 0:
        return self.filter_1_output_0(layer_input, rbf, rij), 0, 0
      elif self.li == 1 and self.lf == 1:
        return self.filter_1_output_1(layer_input, rbf, rij), 0, 0
      else:
        raise NotImplementedError("Other Ls not implemented")

    #layer_input, rbf, rij = inputs
    if self.batch_mode:
      return tf.map_fn(process_row, inputs)[0]
    else:
      return process_row(inputs)[0]

  def R(self, inputs):
    hidden_layer = self.nonlin(self.b1 + tf.tensordot(inputs, self.w1, [[2], [1]]))
    radial = self.b2 + tf.tensordot(hidden_layer, self.w2, [[2], [1]])
    return radial

  def unit_vectors(self, v, axis=-1):
    return v / utils.norm_with_epsilon(v, axis=axis, keep_dims=True)

  def Y_2(self, rij):
    # rij : [N, N, 3]
    # x, y, z : [N, N]
    x = rij[:, :, 0]
    y = rij[:, :, 1]
    z = rij[:, :, 2]
    r2 = tf.maximum(tf.reduce_sum(tf.square(rij), axis=-1), EPSILON)
    # return : [N, N, 5]
    output = tf.stack([x * y / r2,
                      y * z / r2,
                      (-tf.square(x) - tf.square(y) + 2. * tf.square(z)) / (2 * sqrt(3) * r2),
                      z * x / r2,
                      (tf.square(x) - tf.square(y)) / (2. * r2)],
                      axis=-1)
    return output


  def F_0(self, inputs):
    return tf.expand_dims(self.R(inputs), axis=-1)

  def F_1(self, inputs, rij):
    # [N, N, output_dim]
    radial = self.R(inputs)
    # Mask out for dij = 0
    dij = tf.norm(rij, axis=-1)
    condition = tf.tile(tf.expand_dims(dij < EPSILON, axis=-1), [1, 1, self.output_dim])
    masked_radial = tf.where(condition, tf.zeros_like(radial), radial)
    # [N, N, output_dim, 3]
    return tf.expand_dims(self.unit_vectors(rij), axis=-2) * tf.expand_dims(masked_radial, axis=-1)


  def filter_0(self, layer_input, rbf_inputs):
    # [N, N, output_dim, 1]
    F_0_out = self.F_0(rbf_inputs)
    # [N, output_dim]
    # Expand filter axis "j"
    cg = tf.expand_dims(tf.eye(self.input_dim), axis=-2)
    # L x 0 -> L
    return tf.einsum('ijk,abfj,bfk->afi', cg, F_0_out, layer_input)


  def filter_1_output_0(self, layer_input, rbf_inputs, rij):
    F_1_out = self.F_1(rbf_inputs, rij)
    # [N, output_dim, 3]
    if self.input_dim == 1:
        raise ValueError("0 x 1 cannot yield 0")
    elif self.input_dim == 3:
        # 1 x 1 -> 0
        cg = tf.expand_dims(tf.eye(3), axis=0)
        return tf.einsum('ijk,abfj,bfk->afi', cg, F_1_out, layer_input)
    else:
        raise NotImplementedError("Other Ls not implemented")


  def filter_1_output_1(self, layer_input, rbf_inputs, rij):
    # [N, N, output_dim, 3]
    F_1_out = self.F_1(rbf_inputs, rij)
    # [N, output_dim, 3]
    if self.input_dim == 1:
        # 0 x 1 -> 1
        cg = tf.expand_dims(tf.eye(3), axis=-1)
        return tf.einsum('ijk,abfj,bfk->afi', cg, F_1_out, layer_input)
    elif self.input_dim == 3:
        # 1 x 1 -> 1
        return tf.einsum('ijk,abfj,bfk->afi', self.eijk, F_1_out, layer_input)
    else:
        raise NotImplementedError("Other Ls not implemented")

  def get_eijk(self):
    """
    Constant Levi-Civita tensor

    Returns:
        tf.Tensor of shape [3, 3, 3]
    """
    eijk_ = np.zeros((3, 3, 3))
    eijk_[0, 1, 2] = eijk_[1, 2, 0] = eijk_[2, 0, 1] = 1.
    eijk_[0, 2, 1] = eijk_[2, 1, 0] = eijk_[1, 0, 2] = -1.
    return tf.constant(eijk_, dtype=FLOAT_TYPE)

class Convolution(tf.keras.layers.Layer):
  def __init__(self,  **kwargs):
    super(Convolution, self).__init__(**kwargs)

  def build(self, input_shapes):
    n = 0
    self.filters = []
    for key in input_shapes[-3]:
        for i, shape in enumerate(input_shapes[-3][key]):
          self.filters.append(Filter(0))
          n += 1
          if key == 1:
            self.filters.append(Filter(1, 0))
            n += 1
          if key == 0 or key == 1:
            self.filters.append(Filter(1, 1))
            n += 1

  @tf.function
  def call(self, input):
    input_tensor_list, rbf, unit_vectors = input
    n = 0
    output_tensor_list = {0: [], 1: []}
    for key in input_tensor_list:
      for i, tensor in enumerate(input_tensor_list[key]):
        #tensor = tf.identity(tensor, name="in_tensor")
        if True:
          # L x 0 -> L
          tensor_out = self.filters[n]((tensor, rbf, unit_vectors))
          n += 1
          m = 0 if tensor_out.get_shape().as_list()[-1] == 1 else 1
          #tensor_out = tf.identity(tensor_out, name="F0_to_L_out_tensor")
          output_tensor_list[m].append(tensor_out)
        if key == 1:
          # L x 1 -> 0
          tensor_out = self.filters[n]((tensor, rbf, unit_vectors))
          n += 1
          m = 0 if tensor_out.get_shape().as_list()[-1] == 1 else 1
          #tensor_out = tf.identity(tensor_out, name="F1_to_0_out_tensor")
          output_tensor_list[m].append(tensor_out)
        if key == 0 or key == 1:
          # L x 1 -> 1
          tensor_out = self.filters[n]((tensor, rbf, unit_vectors))
          n += 1
          m = 0 if tensor_out.get_shape().as_list()[-1] == 1 else 1
          #tensor_out = tf.identity(tensor_out, name="F1_to_1_out_tensor")
          output_tensor_list[m].append(tensor_out)
    return output_tensor_list
