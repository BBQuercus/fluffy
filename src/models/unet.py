import math
from tensorflow import keras


def encoder_unet(input_layer, depth=3, width=16):
    '''
    '''
    assert (width & (width - 1)) == 0, 'Must be power of 2'

    option_dict_conv = {'kernel_size': (3, 3), 'activation': 'relu', 'padding': 'same'}
    initial_width = int(math.log(width, 2))

    x = input_layer
    y = []
    for i in range(depth):
        w = 2**(i+initial_width)
        x = keras.layers.Conv2D(w, **option_dict_conv)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Conv2D(w, **option_dict_conv)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.MaxPool2D((2, 2))(x)
        y.append(x)

    return y


def decoder_unet(down_layers):
    '''
    '''

    option_dict_conv = {'kernel_size': (3, 3), 'activation': 'relu', 'padding': 'same'}

    x = down_layers[-1]
    y = []
    for d in down_layers[::-1]:
        w = d.shape[-1]
        x = keras.layers.Conv2D(w, **option_dict_conv)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Conv2D(w, **option_dict_conv)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.concatenate([x, d])
        x = keras.layers.UpSampling2D()(x)
        y.append(x)

    return y


def model_seg(**kwargs):
    '''
    '''
    img_size = kwargs.get('img_size', 256)
    depth = kwargs.get('depth', 3)
    n_classes = kwargs.get('n_classes', 3)

    x = keras.layers.Input((img_size, img_size, 1))

    down = encoder_unet(x, depth=depth)
    up = decoder_unet(down)

    option_dict_conv = {'kernel_size': (3, 3), 'activation': 'relu', 'padding': 'same'}
    filters = n_classes if n_classes else 1
    activation = 'softmax' if n_classes else 'sigmoid'

    y = keras.layers.Conv2D(8, **option_dict_conv)(up[-1])
    y = keras.layers.Conv2D(8, **option_dict_conv)(y)
    y = keras.layers.Conv2D(filters, (1, 1), activation=activation)(y)

    model = keras.models.Model(inputs=[x], outputs=[y])

    return model
