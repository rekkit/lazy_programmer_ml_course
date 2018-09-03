import keras.backend as K
from keras.models import Sequential
from keras.applications.vgg16 import VGG16
from keras.layers.convolutional import Conv2D

def truncate_vgg16(input_shape, n_conv_layers):
    if n_conv_layers < 1 or n_conv_layers > 13:
        raise ValueError("The VGG 16 model can be truncated at layers 1 through 13. You requested the model to be "
                         "truncated at layer {0}".format(n_conv_layers))

    # get the pre-trained model
    vgg16 = VGG16(input_shape=input_shape, include_top=False)

    # create a sequential model that is going to contain the requested layers from the VGG16 model and add layers
    i = 0
    model = Sequential()

    for layer in vgg16.layers:
        model.add(layer)

        if layer.__class__ == Conv2D:
            i += 1

            if i > n_conv_layers:
                break

    return model
