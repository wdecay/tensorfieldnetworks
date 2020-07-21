import tensorflow as tf
from dataset import get_dataset
import layers
from ShapeClassificationModel import ShapeClassificationModel

train_dataset, num_classes = get_dataset()

def make_model(layer_dims = [1, 4, 4, 4]):
    shape_input = tf.keras.Input(shape=(4, 3), dtype=tf.float32, batch_size = 5)

    shape_input = tf.convert_to_tensor(list(train_dataset.take(1).as_numpy_iterator())[0][0])

    embed_layer = layers.SelfInteractionSimple(layer_dims[0])
    input_layer = layers.Input()
    
    model_layers = []
    for dim in layer_dims[1:]:
        model_layers.append(layers.Convolution())
        model_layers.append(layers.Concatenation())
        model_layers.append(layers.SelfInteraction(dim))
        model_layers.append(layers.Nonlinearity())
    output_layer = layers.Output(num_classes)

    #x  = embed_layer(tf.ones(shape=(5, 4, 1, 1)))
    x  = embed_layer(tf.ones(shape=(4, 1, 1)))
        
    input_tensor_list = {0: [x]}
    
    rbf, rij = input_layer(shape_input)

    for layer in model_layers:
        if isinstance(layer, layers.Convolution):
            input_tensor_list = layer([input_tensor_list, rbf, rij])
        else:
            input_tensor_list = layer(input_tensor_list)
    output = output_layer(input_tensor_list)

    return tf.keras.Model(inputs = shape_input, outputs = output)


# model = make_model()

# training
model = ShapeClassificationModel(num_classes)
optimizer = tf.keras.optimizers.Adam(learning_rate=1.e-3)

shape_input1 = tf.keras.Input(shape=(4, 3), dtype=tf.float32)

model.compile(optimizer=optimizer,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(), # softmax is applied in the outer
                                                                    # layer (see comment there).
              metrics=['accuracy'])
model.fit(train_dataset.repeat(10), epochs=1)
model.save_weights('model_weights.h5')
