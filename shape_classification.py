#!/usr/bin/env python
# coding: utf-8

# Tensor Field Networks
# 
# Implementation of shape classification demonstration

# In[18]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[19]:


get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import tensorflow.compat.v1 as tf
import random
from math import pi, sqrt
import tensorfieldnetworks.layers as layers
import tensorfieldnetworks.utils as utils
from tensorfieldnetworks.utils import FLOAT_TYPE

tetris = [[(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 1, 0)],  # chiral_shape_1
          [(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, -1, 0)], # chiral_shape_2
          [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)],  # square
          [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3)],  # line
          [(0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0)],  # corner
          [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 0)],  # T
          [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 1)],  # zigzag
          [(0, 0, 0), (1, 0, 0), (1, 1, 0), (2, 1, 0)]]  # L

dataset = [np.array(points_) for points_ in tetris]
num_classes = len(dataset)

tf.disable_eager_execution()


# In[20]:


# radial basis functions
rbf_low = 0.0
rbf_high = 3.5
rbf_count = 4
rbf_spacing = (rbf_high - rbf_low) / rbf_count
centers = tf.cast(tf.lin_space(rbf_low, rbf_high, rbf_count), FLOAT_TYPE)


# In[23]:


# r : [N, 3]
r = tf.placeholder(FLOAT_TYPE, shape=(4, 3))

# rij : [N, N, 3]
rij = utils.difference_matrix(r)

# dij : [N, N]
dij = utils.distance_matrix(r)

# rbf : [N, N, rbf_count]
gamma = 1. / rbf_spacing
rbf = tf.exp(-gamma * tf.square(tf.expand_dims(dij, axis=-1) - centers))

layer_dims = [1, 4, 4, 4]
num_layers = len(layer_dims) - 1

# embed : [N, layer1_dim, 1]
with tf.variable_scope(None, "embed"):
    embed = layers.self_interaction_layer_without_biases(tf.ones(shape=(4, 1, 1)), layer_dims[0])

input_tensor_list = {0: [embed]}

for layer, layer_dim in enumerate(layer_dims[1:]):
    with tf.variable_scope(None, 'layer' + str(layer), values=[input_tensor_list]):
        input_tensor_list = layers.convolution(input_tensor_list, rbf, rij)
        input_tensor_list = layers.concatenation(input_tensor_list)
        input_tensor_list = layers.self_interaction(input_tensor_list, layer_dim)
        input_tensor_list = layers.nonlinearity(input_tensor_list)

tfn_scalars = input_tensor_list[0][0]
tfn_output_shape = tfn_scalars.get_shape().as_list()
tfn_output = tf.reduce_mean(tf.squeeze(tfn_scalars), axis=0)
fully_connected_layer = tf.get_variable('fully_connected_weights', 
                                        [tfn_output_shape[-2], len(dataset)], dtype=FLOAT_TYPE)
output_biases = tf.get_variable('output_biases', [len(dataset)], dtype=FLOAT_TYPE)

# output : [num_classes]
output = tf.einsum('xy,x->y', fully_connected_layer, tfn_output) + output_biases

tf_label = tf.placeholder(tf.int32)

# truth : [num_classes]
truth = tf.one_hot(tf_label, num_classes)

# loss : []
loss = tf.nn.softmax_cross_entropy_with_logits(labels=truth, logits=output)

optim = tf.train.AdamOptimizer(learning_rate=1.e-3)

train_op = optim.minimize(loss)


# In[25]:


max_epochs = 2001
print_freq = 100

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# training
for epoch in range(max_epochs):    
    loss_sum = 0.
    for label, shape in enumerate(dataset):
        loss_value, _ = sess.run([loss, train_op], feed_dict={r: shape, tf_label: label})
        loss_sum += loss_value
        
    if epoch % print_freq == 0:
        print("Epoch %d: validation loss = %.3f" % (epoch, loss_sum / len(dataset)))


# In[26]:


rng = np.random.RandomState()
test_set_size = 25
predictions = [list() for i in range(len(dataset))]

correct_predictions = 0
total_predictions = 0
for i in range(test_set_size):
    for label, shape in enumerate(dataset):
        rotation = utils.random_rotation_matrix(rng)
        rotated_shape = np.dot(shape, rotation)
        translation = np.expand_dims(np.random.uniform(low=-3., high=3., size=(3)), axis=0)
        translated_shape = rotated_shape + translation
        output_label = sess.run(tf.argmax(output), 
                                feed_dict={r: rotated_shape, tf_label: label})
        total_predictions += 1
        if output_label == label:
            correct_predictions += 1
print('Test accuracy: %f' % (float(correct_predictions) / total_predictions))


# In[ ]:




