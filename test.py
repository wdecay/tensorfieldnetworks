import tensorflow as tf
from dataset import get_dataset
import layers
from ShapeClassificationModel import ShapeClassificationModel

train_dataset, num_classes = get_dataset()

def build_model(layer_dims = [1, 4, 4, 4],                
                shape_input = tf.keras.Input(shape=(4, 3), dtype=tf.float32)):

    embed_layer = layers.SelfInteractionSimple(layer_dims[0])
    input_layer = layers.Input()
    
    model_layers = []
    for dim in layer_dims[1:]:
        model_layers.append(layers.Convolution())
        model_layers.append(layers.Concatenation())
        model_layers.append(layers.SelfInteraction(dim))
        model_layers.append(layers.Nonlinearity())
    output_layer = layers.Output(num_classes)

    x, rbf, rij = input_layer(shape_input)
    input_tensor_list = {0: [embed_layer(x)]}
    
    for layer in model_layers:
        if isinstance(layer, layers.Convolution):
            input_tensor_list = layer([input_tensor_list, rbf, rij])
        else:
            input_tensor_list = layer(input_tensor_list)
    output = output_layer(input_tensor_list)

    return tf.keras.Model(inputs = shape_input, outputs = output)


test_ds = train_dataset.map(lambda x, y: x).batch(8).take(1).as_numpy_iterator().next()
model = build_model(
    #shape_input = test_ds
)
model.summary()

optimizer = tf.keras.optimizers.Adam(learning_rate=1.e-3)
@tf.function
def loss_fn(truth, pred):
    return tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels = tf.cast(tf.squeeze(truth), tf.int32),
        logits = pred))

model.compile(optimizer=optimizer, loss = loss_fn, metrics=['accuracy'])
model.fit(train_dataset.repeat(30).batch(8), epochs=4, shuffle=True)

model.save('model_weights.h5')
