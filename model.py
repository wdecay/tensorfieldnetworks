import tensorflow as tf
import layers


def compose_layers(num_classes, layer_dims, shape_input):
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
    return output


def build_model(num_classes,
                layer_dims = [1, 4, 4, 4],
                shape_input = tf.keras.Input(shape=(4, 3), dtype=tf.float32)):

    output = compose_layers(num_classes, layer_dims, shape_input)
    model = tf.keras.Model(inputs = shape_input, outputs = output)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1.e-3)
    @tf.function
    def loss_fn(truth, pred):
        return tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels = tf.cast(tf.squeeze(truth), tf.int32),
            logits = pred))

    model.compile(optimizer=optimizer, loss = loss_fn, metrics=['accuracy'])
    return model


# test with eager execution
if __name__ == "__main__":
    from dataset import get_dataset
    dataset, num_classes = get_dataset()
    test_data = dataset.map(lambda x, y: x).batch(1).take(1).as_numpy_iterator().next()
    print("Input:")
    print(test_data)
    
    result = compose_layers(num_classes, [1, 4, 4, 4], test_data)
    print("Output:")
    print(result)
