import tensorflow as tf
from dataset import get_dataset
from ShapeClassificationModel import ShapeClassificationModel

train_dataset, num_classes = get_dataset()

# training
model = ShapeClassificationModel(num_classes)
optimizer = tf.keras.optimizers.Adam(learning_rate=1.e-3)
model.compile(optimizer=optimizer,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(), # softmax is applied in the outer
                                                                    # layer (see comment there).
              metrics=['accuracy'])
model.fit(train_dataset.repeat(10), epochs=1)
model.save_weights('model_weights.h5')
