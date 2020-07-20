import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import tensorflow as tf
import random
from math import pi, sqrt
import tensorfieldnetworks.utils as utils

from tensorfieldnetworks.ShapeClassificationModel import ShapeClassificationModel

tetris = [[(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 1, 0)],  # chiral_shape_1
          [(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, -1, 0)], # chiral_shape_2
          [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)],  # square
          [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3)],  # line
          [(0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0)],  # corner
          [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 0)],  # T
          [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 1)],  # zigzag
          [(0, 0, 0), (1, 0, 0), (1, 1, 0), (2, 1, 0)]]  # L

dataset = [np.array(points_, dtype='float32') for points_ in tetris]
num_classes = len(dataset)

tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

model = ShapeClassificationModel(num_classes)
model(dataset[0])
model.summary()

max_epochs = 200
print_freq = 10

optimizer = tf.keras.optimizers.Adam(learning_rate=1.e-3)
#with tf.profiler.experimental.Profile('logdir'):
for epoch in range(max_epochs):    
    #loss_sum = 0.
    for label, shape in enumerate(dataset):
        with tf.GradientTape() as tape:
            pred = model(shape, training=True)
            truth = tf.one_hot(label, num_classes)
            loss = tf.nn.softmax_cross_entropy_with_logits(labels=truth, logits=pred)
            loss_sum += loss
        #grads = tape.gradient(loss, model.trainable_variables)
        #optimizer.apply_gradients(grads_and_vars=zip(grads, model.trainable_variables))
        
    if epoch % print_freq == 0:
        print("Epoch %d: validation loss = %.3f" % (epoch, loss_sum / num_classes))

