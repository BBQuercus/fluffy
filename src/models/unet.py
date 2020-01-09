import math
import tensorflow as tf
from tensorflow import keras

def encoder_unet(input_layer, depth=3, width=8, activation='relu', momentum=0.9):
    '''
    '''
    assert (width & (width - 1)) == 0, 'Must be power of 2'

    option_dict_conv = {'kernel_size': (3, 3), 'activation': activation, 'padding': 'same'}
    option_dict_bn = {'axis': -1, 'momentum': momentum}

    initial_width = int(math.log(width, 2))

    y = []
    for i in range(depth):
        w = 2**(i+initial_width)
        if i == 0:
            x = keras.layers.Conv2D(w, **option_dict_conv) (input_layer)
        else:
            x = keras.layers.Conv2D(w, **option_dict_conv) (x)
        x = keras.layers.BatchNormalization(**option_dict_bn) (x)
        x = keras.layers.Conv2D(w, **option_dict_conv) (x)
        x = keras.layers.BatchNormalization(**option_dict_bn) (x)
        x = keras.layers.MaxPool2D((2, 2)) (x)
        y.append(x)
    
    return y

def decoder_unet(down_layers, input_layer=None, activation='relu', momentum=0.9):
    '''
    '''

    option_dict_conv = {'kernel_size': (3, 3), 'activation': activation, 'padding': 'same'}
    option_dict_bn = {'axis': -1, 'momentum': momentum}

    input_layer = down_layers[-1] if not input_layer else input_layer
    y = []

    for i, d in enumerate(down_layers[::-1]):
        w = d.shape[-1]
        if i == 0:
            x = tf.keras.layers.Conv2D(w, **option_dict_conv) (input_layer)
        else:
            x = tf.keras.layers.Conv2D(w, **option_dict_conv) (x)
        x = tf.keras.layers.BatchNormalization(**option_dict_bn) (x)
        x = tf.keras.layers.Conv2D(w, **option_dict_conv) (x)
        x = tf.keras.layers.BatchNormalization(**option_dict_bn) (x)
        x = tf.keras.layers.UpSampling2D() (x)
        x = tf.keras.layers.concatenate([x, d], axis=3)
        y.append(x)
    
    return y

def model_segmentation(depth=3, width=8, img_size=None, categories=None, activation='relu', momentum=0.9):
    '''
    '''
    option_dict_conv = {'kernel_size': (3, 3), 'activation': activation, 'padding': 'same'}

    x = tf.keras.layers.Input((img_size, img_size, 1))

    down = encoder_unet(x, depth=depth, width=width)
    up = decoder_unet(down)

    y = tf.keras.layers.Conv2D(8, **option_dict_conv) (up[-1])
    y = tf.keras.layers.Conv2D(8, **option_dict_conv) (y)

    filters = categories if categories else 1
    activation = 'softmax' if categories else 'sigmoid'

    y = tf.keras.layers.Conv2D(filters, (1, 1), activation=activation) (y)

    model = tf.keras.models.Model(inputs=[x], outputs=[y])

    return model