import random
import glob
import skimage
import datetime
import numpy as np
import tensorflow as tf


def train_valid_split(x_list, y_list, valid_split=0.25):
    '''
    Splits two lists (images and masks) into random training and validation sets.

    Args:
        - x_list (list): List containing filenames of all images.
        - y_list (list): List containing filenames of all masks.
        - valid_split (float, optional): Number between 0-1 to denote
            the percentage of examples used for validation.
    Returns:
        - x_train, x_valid, y_train, y_valid (lists): Splited lists
            containing training or validation examples respectively.
    '''

    assert len(x_list) == len(y_list), 'Inputs must be of same length.'
    assert len(x_list) != 0, 'Inputs must contain values.'
    assert 0 <= valid_split <= 1, 'Split must be between 0 and 1.'

    # Random shuffle
    combined = list(zip(x_list, y_list))
    random.shuffle(combined)
    x_list, y_list = zip(*combined)

    # Split into train / valid
    split_len = round(len(x_list) * valid_split)
    x_valid, x_train = x_list[:split_len], x_list[split_len:]
    y_valid, y_train = y_list[:split_len], y_list[split_len:]

    return x_train, x_valid, y_train, y_valid


def standard_unet(img_size=None):
    '''
    Builds a UNet model for binary segmentation of gray-scale images.

    Args:
        - img_size (int, optional): Image size used to create the model,
            must be a power of two and greater than 32.
    Returns:
        - model (tf.keras.models.Model): UNet model for segmentation.
    '''

    assert img_size in (2 ** np.arange(6, 20)), 'Size must be power of two and >32.'

    option_dict_conv = {'activation': 'relu', 'padding': 'same'}
    option_dict_bn = {'axis': -1, 'momentum': 0.9}

    x = tf.keras.layers.Input((img_size, img_size, 1))

    # Down
    a = tf.keras.layers.Conv2D(16, (3, 3), **option_dict_conv)(x)
    a = tf.keras.layers.BatchNormalization(**option_dict_bn)(a)
    a = tf.keras.layers.Conv2D(16, (3, 3), **option_dict_conv)(a)
    a = tf.keras.layers.BatchNormalization(**option_dict_bn)(a)
    a = tf.keras.layers.Dropout(0.1)(a)
    y = tf.keras.layers.MaxPool2D((2, 2))(a)

    b = tf.keras.layers.Conv2D(32, (3, 3), **option_dict_conv)(y)
    b = tf.keras.layers.BatchNormalization(**option_dict_bn)(b)
    b = tf.keras.layers.Conv2D(32, (3, 3), **option_dict_conv)(b)
    b = tf.keras.layers.BatchNormalization(**option_dict_bn)(b)
    b = tf.keras.layers.Dropout(0.2)(b)
    y = tf.keras.layers.MaxPool2D((2, 2))(b)

    c = tf.keras.layers.Conv2D(64, (3, 3), **option_dict_conv)(y)
    c = tf.keras.layers.BatchNormalization(**option_dict_bn)(c)
    c = tf.keras.layers.Conv2D(64, (3, 3), **option_dict_conv)(c)
    c = tf.keras.layers.BatchNormalization(**option_dict_bn)(c)
    c = tf.keras.layers.Dropout(0.3)(c)
    y = tf.keras.layers.MaxPool2D((2, 2))(c)

    d = tf.keras.layers.Conv2D(128, (3, 3), **option_dict_conv)(y)
    d = tf.keras.layers.BatchNormalization(**option_dict_bn)(d)
    d = tf.keras.layers.Conv2D(128, (3, 3), **option_dict_conv)(d)
    d = tf.keras.layers.BatchNormalization(**option_dict_bn)(d)
    d = tf.keras.layers.Dropout(0.4)(d)
    y = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(d)

    # Up
    e = tf.keras.layers.Conv2D(128, (3, 3), **option_dict_conv)(y)
    e = tf.keras.layers.BatchNormalization(**option_dict_bn)(e)
    e = tf.keras.layers.Conv2D(128, (3, 3), **option_dict_conv)(e)
    e = tf.keras.layers.BatchNormalization(**option_dict_bn)(e)
    e = tf.keras.layers.Dropout(0.4)(e)
    e = tf.keras.layers.UpSampling2D()(e)
    y = tf.keras.layers.concatenate([e, d], axis=3)

    f = tf.keras.layers.Conv2D(64, (3, 3), **option_dict_conv)(y)
    f = tf.keras.layers.BatchNormalization(**option_dict_bn)(f)
    f = tf.keras.layers.Conv2D(64, (3, 3), **option_dict_conv)(f)
    f = tf.keras.layers.BatchNormalization(**option_dict_bn)(f)
    f = tf.keras.layers.Dropout(0.3)(f)
    f = tf.keras.layers.UpSampling2D()(f)
    y = tf.keras.layers.concatenate([f, c], axis=3)

    g = tf.keras.layers.Conv2D(32, (3, 3), **option_dict_conv)(y)
    g = tf.keras.layers.BatchNormalization(**option_dict_bn)(g)
    g = tf.keras.layers.Conv2D(32, (3, 3), **option_dict_conv)(g)
    g = tf.keras.layers.BatchNormalization(**option_dict_bn)(g)
    g = tf.keras.layers.Dropout(0.2)(g)
    g = tf.keras.layers.UpSampling2D()(g)
    y = tf.keras.layers.concatenate([g, b], axis=3)

    h = tf.keras.layers.Conv2D(16, (3, 3), **option_dict_conv)(y)
    h = tf.keras.layers.BatchNormalization(**option_dict_bn)(h)
    h = tf.keras.layers.Conv2D(16, (3, 3), **option_dict_conv)(h)
    h = tf.keras.layers.BatchNormalization(**option_dict_bn)(h)
    h = tf.keras.layers.Dropout(0.1)(h)
    h = tf.keras.layers.UpSampling2D()(h)
    y = tf.keras.layers.concatenate([h, a], axis=3)

    y = tf.keras.layers.Conv2D(8, (3, 3), **option_dict_conv)(y)
    y = tf.keras.layers.Conv2D(8, (3, 3), **option_dict_conv)(y)

    y = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(y)

    model = tf.keras.models.Model(inputs=[x], outputs=[y])

    return model


def random_sample_generator(x_list, y_list, batch_size, bit_depth, img_size):
    '''
    Yields a generator for training examples with augmentation.

    Args:
        - x_list (list): List containing filepaths to all usable images.
        - y_list (list): List containing filepaths to all usable masks.
        - batch_size (int): Size of one mini-batch.
        - bit_depth (int): Bit depth of images to normalize.
        - img_size (int): Size to crop images to.
    Returns:
        - Generator: List of augmented training examples of
            size (batch_size, img_size, img_size, 1)
    '''

    assert len(x_list) == len(y_list), 'Lists must be of equal size.'
    assert len(x_list) != 0, 'Lists must contain values.'
    assert type(x_list[0]) == str, 'Lists must contain strings.'
    assert (type(batch_size), type(bit_depth), type(img_size)) == (int, int, int)

    while True:

        # Buffers for a batch of data
        x = np.zeros((batch_size, img_size, img_size, 1))
        y = np.zeros((batch_size, img_size, img_size, 1))

        # Get one image at a time
        for i in range(batch_size):

            # Get random image
            img_index = np.random.randint(low=0, high=len(x_list))

            # Open images and normalize
            x_curr = skimage.io.imread(x_list[img_index]) * (1./(2**bit_depth - 1))
            y_curr = skimage.io.imread(y_list[img_index])

            # Get random crop
            start_dim1 = np.random.randint(low=0, high=x_curr.shape[0] - img_size) if x_curr.shape[0] > img_size else 0
            start_dim2 = np.random.randint(low=0, high=x_curr.shape[1] - img_size) if x_curr.shape[1] > img_size else 0
            patch_x = x_curr[start_dim1:start_dim1 + img_size, start_dim2:start_dim2 + img_size]
            patch_y = y_curr[start_dim1:start_dim1 + img_size, start_dim2:start_dim2 + img_size]

            rand_flip = np.random.randint(low=0, high=2)
            rand_rotate = np.random.randint(low=0, high=4)
            rand_illumination = 1 + np.random.uniform(-0.75, 0.75)

            # Flip
            if rand_flip == 0:
                patch_x = np.flip(patch_x, 0)
                patch_y = np.flip(patch_y, 0)

            # Rotate
            for _ in range(rand_rotate):
                patch_x = np.rot90(patch_x)
                patch_y = np.rot90(patch_y)

            # Illumination
            patch_x *= rand_illumination

            # Save image to buffer
            x[i, :, :, 0] = patch_x
            y[i, :, :, 0] = patch_y

        # Return the buffer
        yield(x, y)


def single_data_from_images(x_list, y_list, batch_size, bit_depth, img_size):
    '''
    Yields a generator for validation examples without augmentation.

    Args:
        - x_list (list): List containing filepaths to all usable images.
        - y_list (list): List containing filepaths to all usable masks.
        - batch_size (int): Size of one mini-batch.
        - bit_depth (int): Bit depth of images to normalize.
        - img_size (int): Size to crop images to.
    Returns:
        - Generator: List of augmented training examples of
            size (batch_size, img_size, img_size, 1)
    '''

    assert len(x_list) == len(y_list), 'Lists must be of equal size.'
    assert len(x_list) != 0, 'Lists must contain values.'
    assert type(x_list[0]) == str, 'Lists must contain strings.'
    assert (type(batch_size), type(bit_depth), type(img_size)) == (int, int, int)

    while True:

        # Buffers for a batch of data
        x = np.zeros((batch_size, img_size, img_size, 1))
        y = np.zeros((batch_size, img_size, img_size, 1))

        # Get one image at a time
        for i in range(batch_size):

            # Get random image
            img_index = np.random.randint(low=0, high=len(x_list))

            # Open images and normalize
            x_curr = skimage.io.imread(x_list[img_index]) * (1./(2**bit_depth - 1))
            y_curr = skimage.io.imread(y_list[img_index])

            # Get random crop
            start_dim1 = np.random.randint(low=0, high=x_curr.shape[0] - img_size) if x_curr.shape[0] > img_size else 0
            start_dim2 = np.random.randint(low=0, high=x_curr.shape[1] - img_size) if x_curr.shape[1] > img_size else 0
            patch_x = x_curr[start_dim1:start_dim1 + img_size, start_dim2:start_dim2 + img_size]
            patch_y = y_curr[start_dim1:start_dim1 + img_size, start_dim2:start_dim2 + img_size]

            # Save image to buffer
            x[i, :, :, 0] = patch_x
            y[i, :, :, 0] = patch_y

        # Return the buffer
        yield(x, y)


def main():
    root = './data/train_val/'
    img_size = 256

    # Import paths
    X = sorted(glob.glob(f'{root}images/*.tif'))
    Y = sorted(glob.glob(f'{root}masks/*.tif'))

    # Train / valid split
    x_train, x_valid, y_train, y_valid = train_valid_split(x_list=X, y_list=Y, valid_split=0.2)

    # Build model
    tf.keras.backend.clear_session()
    model = standard_unet()
    model.summary()

    # Compile model
    loss = tf.keras.losses.binary_crossentropy
    metrics = [tf.keras.metrics.binary_accuracy]
    optimizer = tf.keras.optimizers.Adam(lr=0.0001)

    model.compile(loss=loss, metrics=metrics, optimizer=optimizer)

    # Callbacks
    model_name = f"./models/{datetime.date.today().strftime('%Y%m%d')}_binary"
    callbacks = [tf.keras.callbacks.ModelCheckpoint(f'{model_name}.h5', save_best_only=True),
                 tf.keras.callbacks.CSVLogger(filename=f'{model_name}.csv'),
                 tf.keras.callbacks.TensorBoard(model_name)]

    # Build generators
    train_gen = random_sample_generator(
        x_list=x_train,
        y_list=y_train,
        batch_size=16,
        bit_depth=16,
        img_size=img_size)

    val_gen = single_data_from_images(
        x_valid,
        y_valid,
        batch_size=16,
        bit_depth=16,
        img_size=img_size)

    # Training
    history = model.fit_generator(generator=train_gen,
                                  steps_per_epoch=20,
                                  epochs=250,
                                  validation_data=val_gen,
                                  validation_steps=20,
                                  callbacks=callbacks,
                                  verbose=2)


if __name__ == "__main__":
    main()
