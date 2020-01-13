from tensorflow import keras

import models.resnet50


def resnet50_decoder(down_layers):
    '''
    '''

    x = down_layers[-1]
    x = keras.layers.Conv2D(x.shape[-1], (3, 3), padding='same')(down_layers[-1])
    x = keras.layers.Activation('relu')(x)
    y = [x]

    # TODO allow for depth changes (don't use last layers)
    for d in down_layers[-2::-1]:
        x = keras.layers.Conv2DTranspose(d.shape[-1], (3, 3), strides=(2, 2), padding='same')(x)
        x = keras.layers.concatenate([x, d])
        x = keras.layers.Activation('relu')(x)
        y.append(x)

    # Solve problem here
    x = keras.layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(x)
    y.append(x)

    return y


def model_seg(**kwargs):
    '''
    '''
    img_size = kwargs.get('img_size', 224)
    # depth = kwargs.get('depth', 4)
    categories = kwargs.get('categories', 3)

    # TODO add optional freezing?
    x, down = models.resnet50.resnet50_encoder(img_size)
    up = resnet50_decoder(down)

    option_dict_conv = {'kernel_size': (3, 3), 'padding': 'same'}
    y = keras.layers.Conv2D(8, **option_dict_conv)(up[-1])
    y = keras.layers.Conv2D(8, **option_dict_conv)(y)

    filters = categories if categories else 1
    activation = 'softmax' if categories else 'sigmoid'

    y = keras.layers.Conv2D(filters, (1, 1), activation=activation)(y)

    model = keras.models.Model(inputs=[x], outputs=[y])

    return model
