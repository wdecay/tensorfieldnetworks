import tensorflow as tf

from tensorfieldnetworks.InputLayer import InputLayer
from tensorfieldnetworks.SelfInteractionLayer import SelfInteractionSimple, SelfInteractionLayer
from tensorfieldnetworks.ConvolutionLayer import ConvolutionLayer
from tensorfieldnetworks.ConcatenationLayer import ConcatenationLayer
from tensorfieldnetworks.NonlinearityLayer import NonlinearityLayer
from tensorfieldnetworks.OutputLayer import OutputLayer

class ShapeClassificationModel(tf.keras.Model):
    def __init__(self, num_classes, layer_dims = [1, 4, 4, 4]):
        super(ShapeClassificationModel, self).__init__()
        self.embed_layer = SelfInteractionSimple(layer_dims[0])
        self.input_layer = InputLayer()
        
        self.model_layers = []
        for dim in layer_dims[1:]:
            self.model_layers.append(ConvolutionLayer())
            self.model_layers.append(ConcatenationLayer())
            self.model_layers.append(SelfInteractionLayer(dim))
            self.model_layers.append(NonlinearityLayer())

        self.output_layer = OutputLayer(num_classes)
    
    #@tf.function
    def call(self, input, training=False):
        x  = self.embed_layer(tf.ones(shape=(4, 1, 1)))
        input_tensor_list = {0: [x]}
        rbf, rij = self.input_layer(input, training)
        #print("test")
        for layer in self.model_layers:
            if isinstance(layer, ConvolutionLayer): 
                input_tensor_list = layer([input_tensor_list, rbf, rij])
            else:
                input_tensor_list = layer(input_tensor_list)
        return self.output_layer(input_tensor_list)
        #return tf.one_hot(1, 8)
