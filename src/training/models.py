import logging
import math
import numpy as np
import tensorflow as tf

LOG_FORMAT = "%(levelname)s %(asctime)s - %(filename)s %(funcName)s %(lineno)s - %(message)s"
logging.basicConfig(filename="./train_model.log",
                    level=logging.DEBUG,
                    format=LOG_FORMAT,
                    filemode="a")
log = logging.getLogger()


def unet_encoder(input_layer, depth, width):
    """
    Returns skip connections of the encoder part of the UNet.

    Args:
        - input_layer (tf.keras.layers): Input layer, typically tf.keras.layers.Input().
        - depth (int): How many convolutional blocks the main part of the model should have.
        - model_width (int): Initial number of filters in first convolutional block.
    Returns:
        - y (tf.keras.layer): Output layer of the encoder.
        - skips (list): List of layers to be used as skip connections.
    """
    if not isinstance(width, int):
        raise TypeError(f"width must be str but is {type(width)}.")
    if not (width & (width-1) == 0) and width != 0:
        raise ValueError(f"width must be a power of 2 but is {width}.")

    option_dict_conv = {
        "kernel_size": (3, 3),
        "activation": "relu",
        "padding": "same"
    }
    option_dict_bn = {
        "axis": -1,
        "momentum": 0.9
    }

    drops = np.arange(1, depth+1)*0.1
    drops = np.where(drops > 0.5, 0.5, drops)
    width_log2 = int(math.log2(width))
    widths = 2**np.arange(width_log2, width_log2+depth)

    skips = []
    y = input_layer
    for w, d in zip(widths, drops):
        x = tf.keras.layers.Conv2D(w, **option_dict_conv)(y)
        x = tf.keras.layers.BatchNormalization(**option_dict_bn)(x)
        x = tf.keras.layers.Conv2D(w, **option_dict_conv)(x)
        x = tf.keras.layers.BatchNormalization(**option_dict_bn)(x)
        x = tf.keras.layers.Dropout(d)(x)
        skips.append(x)
        y = tf.keras.layers.MaxPool2D((2, 2))(x)

    return y, skips


def unet_decoder(input_layer, skips):
    """
    Builds the decoder part of the UNet.

    Args:
        - input_layer (tf.keras.layers): Layer to which the decoder connects.
        - skips (list): Skip connections to be used in the decoder.
    Returns:
        - y (tf.keras.layers): Output layer.
    """
    if not isinstance(skips, list):
        raise TypeError(f"skips must be list but is {type(skips)}.")

    option_dict_conv = {
        "kernel_size": (3, 3),
        "activation": "relu",
        "padding": "same"
    }
    option_dict_bn = {
        "axis": -1,
        "momentum": 0.9
    }

    depth = len(skips)
    drops = np.arange(1, depth+1)*0.1
    drops = np.where(drops > 0.5, 0.5, drops)[::-1]
    width = skips[0].shape.as_list()[-1]
    width_log2 = int(math.log2(width))
    widths = 2**np.arange(width_log2, width_log2+depth)[::-1]

    y = input_layer
    for s, w, d in zip(skips[::-1], widths, drops):
        x = tf.keras.layers.Conv2D(w, **option_dict_conv)(y)
        x = tf.keras.layers.BatchNormalization(**option_dict_bn)(x)
        x = tf.keras.layers.Conv2D(w, **option_dict_conv)(x)
        x = tf.keras.layers.BatchNormalization(**option_dict_bn)(x)
        x = tf.keras.layers.Dropout(d)(x)
        x = tf.keras.layers.UpSampling2D()(x)
        y = tf.keras.layers.concatenate([x, s], axis=3)

    return y


# TODO check isinstance of layers - IDK how
def unet(binary=True, input_size=None, depth=4, model_width=16, last_width=16):
    """
    Builds a flexible UNet model for binary or categorical segmentation of gray-scale images.

    Args:
        - binary (bool): If returned model should be binary or categorical.
        - img_size (None / int): Image size used to create the model,
            must be a power of two and greater than 32.
        - depth (int): How many convolutional blocks the main part of the model should have.
        - model_width (int): Initial number of filters in first convolutional block.
        - last_width (int): Number of filters in last convolutional block before output.
    Returns:
        - model (tf.keras.models.Model): UNet model for segmentation.
    """
    if not isinstance(binary, bool):
        raise TypeError(f"binary must be bool but is {type(binary)}.")
    if not all(isinstance(i, int) for i in [depth, model_width, last_width]):
        raise TypeError(f"depth, model_width, last_width must be int but are {[type(i) for i in [depth, model_width, last_width]]}.")
    if ((input_size is not None) and (input_size not in (2 ** np.arange(5, 20)))):
        raise ValueError(f"img_size must be None or a power of 2 and >32 but is {input_size}.")
    tf.keras.backend.clear_session()
    x = tf.keras.layers.Input((input_size, input_size, 1))

    y, skips = unet_encoder(x, depth, model_width)
    y = unet_decoder(y, skips)

    option_dict_conv = {"kernel_size": (3, 3), "activation": "relu", "padding": "same"}
    y = tf.keras.layers.Conv2D(last_width, **option_dict_conv)(y)
    y = tf.keras.layers.Conv2D(last_width, **option_dict_conv)(y)

    if binary:
        channels = 1
        activation = "sigmoid"
    else:
        channels = 3
        activation = "softmax"
    y = tf.keras.layers.Conv2D(channels, (1, 1), activation=activation)(y)

    model = tf.keras.models.Model(inputs=[x], outputs=[y])

    return model


# TODO typecheck history / automate which type of accuracy
def get_accuracy(history):
    """ Returns the latest accuracy metric from the training history. """
    return history.history["val_binary_accuracy"]
    # return history.history["val_categorical_accuracy"][-1]
