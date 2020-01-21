import math
from tensorflow import keras


def encoder_unet(input_layer, depth=3, width=16, momentum=0.9, dropout=0.2):
    '''
    Encoder side of UNet returning the "down" layers.

    Args:
        input_layer (tf.keras.layers): Layer to start with.
        depth (int): Number of conv / maxpool blocks.
        width (int): Starting number of filters, doubles with
            each step into the depth.

    Returns:
        y (list): tf.keras.layers of "down" layers.
    '''
    assert (width & (width - 1)) == 0, 'Must be power of 2'

    option_dict_conv = {'kernel_size': (3, 3), 'activation': 'relu', 'padding': 'same'}
    option_dict_bn = {'momentum': momentum}
    initial_width = int(math.log(width, 2))

    x = input_layer
    y = []
    for i in range(depth):
        w = 2**(i+initial_width)
        x = keras.layers.Conv2D(w, **option_dict_conv)(x)
        x = keras.layers.BatchNormalization(**option_dict_bn)(x)
        x = keras.layers.Dropout(dropout)(x)
        x = keras.layers.Conv2D(w, **option_dict_conv)(x)
        x = keras.layers.BatchNormalization(**option_dict_bn)(x)
        x = keras.layers.Dropout(dropout)(x)
        x = keras.layers.MaxPool2D((2, 2))(x)
        y.append(x)

    return y


def decoder_unet(down_layers, momentum=0.9, dropout=0.2):
    '''
    Decoder side of UNet returning the "up" layers.
    Automatically adjusts sizes to match down layers.

    Args:
        down_layers (list of. tf.keras.layers): Layers yielded
            from function encoder_unet.

    Returns:
        y (list): tf.keras.layers of "up" layers.
    '''

    option_dict_conv = {'kernel_size': (3, 3), 'activation': 'relu', 'padding': 'same'}
    option_dict_bn = {'momentum': momentum}

    x = down_layers[-1]
    y = []
    for d in down_layers[::-1]:
        w = d.shape[-1]
        x = keras.layers.Conv2D(w, **option_dict_conv)(x)
        x = keras.layers.BatchNormalization(**option_dict_bn)(x)
        x = keras.layers.Dropout(dropout)(x)
        x = keras.layers.Conv2D(w, **option_dict_conv)(x)
        x = keras.layers.BatchNormalization(**option_dict_bn)(x)
        x = keras.layers.Dropout(dropout)(x)
        x = keras.layers.concatenate([x, d])
        x = keras.layers.UpSampling2D()(x)
        y.append(x)

    return y


def model_seg(**kwargs):
    '''
    Builds a full UNet model for segmentation. Not yet compiled.

    Args:
        ––– **kwargs:
        img_size (int): Size of input images.
        depth (int): Number of conv / maxpool blocks in backbone.
        width (int): Starting number of filters of the down layers.
            Doubles with each depth.
        classes (bool): If classes used for segmentation (BG, FG, Border).

    Returns:
        model (tf.keras.models.Model): UNet model for semantic segmentation.
    '''
    img_size = kwargs.get('img_size', None)
    depth = kwargs.get('depth', 3)
    width = kwargs.get('width', 16)
    momentum = kwargs.get('momentum', 0.9)
    dropout = kwargs.get('dropout', 0.2)
    classes = kwargs.get('classes', 4)

    x = keras.layers.Input((img_size, img_size, 1))

    down = encoder_unet(x, depth=depth, width=width, momentum=momentum, dropout=dropout)
    up = decoder_unet(down, momentum=momentum, dropout=dropout)

    # Keep activation relu?
    option_dict_conv = {'kernel_size': (3, 3), 'padding': 'same'}
    y = keras.layers.Conv2D(width, **option_dict_conv)(up[-1])
    y = keras.layers.Conv2D(width, **option_dict_conv)(y)

    activation = 'softmax' if classes > 1 else 'sigmoid'
    y = keras.layers.Conv2D(classes, (1, 1), activation=activation)(y)

    model = keras.models.Model(inputs=[x], outputs=[y])

    return model
