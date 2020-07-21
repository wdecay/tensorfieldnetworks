import tensorflow as tf
import layers

class ShapeClassificationModel(tf.keras.Model):
    def __init__(self, num_classes, layer_dims = [1, 4, 4, 4]):
        super().__init__()
        self.embed_layer = layers.SelfInteractionSimple(layer_dims[0])
        self.input_layer = layers.Input()
        
        self.model_layers = []
        for dim in layer_dims[1:]:
            self.model_layers.append(layers.Convolution())
            self.model_layers.append(layers.Concatenation())
            self.model_layers.append(layers.SelfInteraction(dim))
            self.model_layers.append(layers.Nonlinearity())

        self.output_layer = layers.Output(num_classes)
    
    def call(self, input, training=False):
        x  = self.embed_layer(tf.ones(shape=(4, 1, 1)))
        input_tensor_list = {0: [x]}
        rbf, rij = self.input_layer(input)

        for layer in self.model_layers:
            if isinstance(layer, layers.Convolution): 
                input_tensor_list = layer([input_tensor_list, rbf, rij])
            else:
                input_tensor_list = layer(input_tensor_list)
        
        return self.output_layer(input_tensor_list)
