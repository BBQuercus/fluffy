import logging
import glob
import skimage
import datetime
import numpy as np
import tensorflow as tf
import scipy.ndimage as ndi

LOG_FORMAT = '%(levelname)s %(asctime)s - %(filename)s %(funcName)s %(lineno)s - %(message)s'
logging.basicConfig(filename='./logs/train_model.log',
                    level=logging.DEBUG,
                    format=LOG_FORMAT,
                    filemode='a')
log = logging.getLogger()


def shuffle_lists(list_1, list_2):
    ''' Shuffles two lists keeping items in the same relative locations. '''
    if not all(isinstance(i, list) for i in [list_1, list_2]):
        raise TypeError(f'x_list, y_list must be list but is {type(list_2)}, {type(list_2)}')

    import random
    combined = list(zip(list_1, list_2))
    random.shuffle(combined)
    tup_1, tup_2 = zip(*combined)
    return list(tup_1), list(tup_2)


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
    if not all(isinstance(i, list) for i in [x_list, y_list]):
        raise TypeError(f'x_list, y_list must be list but is {type(x_list)}, {type(y_list)}')
    if not any(isinstance(valid_split, i) for i in [int, float]):
        raise TypeError(f'valid_split must be float but is a {type(valid_split)}')
    if not all(isinstance(i, str) for i in x_list):
        raise TypeError(f'x_list items must be str but are {type(x_list[0])}')
    if not all(isinstance(i, str) for i in x_list):
        raise TypeError(f'x_list items must be str but are {type(x_list[0])}')
    if len(x_list) != len(y_list):
        raise ValueError(f'Lists must be of equal length: {len(x_list)} != {len(y_list)}')
    if any(len(i) == 0 for i in x_list):
        raise ValueError('x_list items must not be empty')
    if any(len(i) == 0 for i in y_list):
        raise ValueError('y_list items must not be empty')
    if len(x_list) <= 2:
        raise ValueError('Lists must contain 2 elements or more')
    if not 0 <= valid_split <= 1:
        raise ValueError(f'valid_split must be between 0-1 but is {valid_split}')

    split_len = round(len(x_list) * valid_split)

    x_list, y_list = shuffle_lists(x_list, y_list)
    x_valid, x_train = x_list[:split_len], x_list[split_len:]
    y_valid, y_train = y_list[:split_len], y_list[split_len:]

    return x_train, x_valid, y_train, y_valid


def standard_unet(input_size=None):
    '''
    Builds a UNet model for categorical segmentation of gray-scale images.

    Args:
        - img_size (int, optional): Image size used to create the model,
            must be a power of two and greater than 32.
    Returns:
        - model (tf.keras.models.Model): UNet model for segmentation.
    '''
    if ((input_size is not None) and (input_size not in (2 ** np.arange(5, 20)))):
        raise ValueError(f'img_size must be None or a power of 2 and >32 but is {input_size}')

    option_dict_conv = {'activation': 'relu', 'padding': 'same'}
    option_dict_bn = {'axis': -1, 'momentum': 0.9}

    tf.keras.backend.clear_session()
    x = tf.keras.layers.Input((input_size, input_size, 1))

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

    y = tf.keras.layers.Conv2D(16, (3, 3), **option_dict_conv)(y)
    y = tf.keras.layers.Conv2D(16, (3, 3), **option_dict_conv)(y)

    y = tf.keras.layers.Conv2D(3, (1, 1), activation='softmax')(y)

    model = tf.keras.models.Model(inputs=[x], outputs=[y])

    return model


def add_borders(mask, border_size=2):
    '''
    Adds borders to labeled masks.

    Args:
        - mask (np.ndarray): Mask with uniquely labeled
            object to which borders will be added.
        - border_size (int): Size of border in pixels.
    Returns:
        - output_mask (np.ndarray): Mask with three channels –
            Background (0), Objects (1), and Borders (2).
    '''
    if not isinstance(mask, np.ndarray):
        raise TypeError(f'input_mask must be a np.ndarray but is a {type(mask)}')
    if not isinstance(border_size, int):
        raise TypeError(f'size must be an int but is a {type(border_size)}')

    borders = np.zeros(mask.shape)
    for i in np.unique(mask):
        curr_mask = np.where(mask == i, 1, 0)
        mask_dil = ndi.morphology.binary_dilation(curr_mask, iterations=border_size)
        mask_ero = ndi.morphology.binary_erosion(curr_mask, iterations=border_size)
        mask_border = np.logical_xor(mask_dil, mask_ero)
        borders[mask_border] = i
    output_mask = np.where(borders > 0, 2, mask > 0)
    return output_mask


def add_augmentation(image, mask):
    '''
    Adds random augmentation to image and mask.

    Args:
        - image (np.ndarray): Image to be augmented.
        - mask (np.ndarray): Mask to be augmented.
    Returns:
        image, mask (np.ndarray): Augmented image and mask respectively.
    '''
    if not all(isinstance(i, np.ndarray) for i in [image, mask]):
        raise TypeError(f'image, mask must be a np.ndarray but are {type(image)}, {type(mask)}')

    rand_flip = np.random.randint(low=0, high=2)
    rand_rotate = np.random.randint(low=0, high=4)
    rand_illumination = 1 + np.random.uniform(-0.75, 0.75)

    # Flipping
    if rand_flip == 0:
        image = np.flip(image, 0)
        mask = np.flip(mask, 0)

    # Rotation
    for _ in range(rand_rotate):
        image = np.rot90(image)
        mask = np.rot90(mask)

    # Illumination
    image = np.multiply(image, rand_illumination)

    return image, mask


def random_cropping(image, mask, crop_size):
    '''
    Randomly crops an image and mask to size crop_size.
    
    Args:
        - image (np.ndarray): Image to be cropped.
        - mask (np.ndarray): Mask to be cropped.
        - crop_size (int): Size to crop image and mask (both dimensions).
    
    Returns:
        - crop_image, crop_mask (np.ndarray): Cropped image and mask
            respectively with size crop_size x crop_size.
    '''
    if not all(isinstance(i, np.ndarray) for i in [image, mask]):
        raise TypeError(f'image, mask must be np.ndarray but is {type(image)}, {type(mask)}')
    if not isinstance(crop_size, int):
        raise TypeError(f'crop_size must be an int but is {type(crop_size)}')
    if not image.shape[:2] == mask.shape[:2]:
        raise ValueError(f'image, mask must be of same shape: {image.shape[:2]} != {mask.shape[:2]}')
    if crop_size == 0:
        raise ValueError(f'crop_size must be larger than 0')

    start_dim1 = np.random.randint(low=0, high=image.shape[0] - crop_size) if image.shape[0] > crop_size else 0
    start_dim2 = np.random.randint(low=0, high=image.shape[1] - crop_size) if image.shape[1] > crop_size else 0

    crop_image = image[start_dim1:start_dim1 + crop_size, start_dim2:start_dim2 + crop_size]
    crop_mask = mask[start_dim1:start_dim1 + crop_size, start_dim2:start_dim2 + crop_size]

    return crop_image, crop_mask


def normalize_image(image, bit_depth=16):
    ''' Normalizes image by bit_depth. '''
    if not isinstance(image, np.ndarray):
        raise TypeError(f'image must be np.ndarray but is {type(image)}')
    if not isinstance(bit_depth, int):
        raise TypeError(f'bit_depth must be int but is {type(int)}')
    if bit_depth == 0:
        raise ZeroDivisionError(f'bit_depth must not be zero')

    return image * (1./(2**bit_depth - 1))


# TODO refactor to simpler function
def random_sample_generator(x_list, y_list, augment, batch_size, bit_depth, crop_size):
    '''
    Yields a generator for training examples with augmentation.

    Args:
        - x_list (list): List containing filepaths to all usable images.
        - y_list (list): List containing filepaths to all usable masks.
        - augment (bool): If augmentation should be done.
        - batch_size (int): Size of one mini-batch.
        - bit_depth (int): Bit depth of images to normalize.
        - crop_size (int): Size to crop images to.
    Returns:
        - Generator: List of augmented training examples of
            size (batch_size, img_size, img_size, 3)
    '''
    if not all(isinstance(i, list) for i in [x_list, y_list]):
        raise TypeError(f'x_list, y_list must be list but are {type(x_list)}, {type(y_list)}')
    if not all(isinstance(y, str) for y in y_list):
        raise TypeError(f'Elements of y_list must be a str')
    if not isinstance(augment, bool):
        raise TypeError(f'augment must be bool but is {type(bool)}')
    if not all(isinstance(i, int) for i in [batch_size, bit_depth, crop_size]):
        raise TypeError(f'batch_size, bit_depth, crop_size must be int but are {type(batch_size)}, {type(bit_depth)}, {type(crop_size)}')
    if len(x_list) != len(y_list):
        raise ValueError(f'Lists must be of equal length: {len(x_list)} != {len(y_list)}')
    if len(x_list) < 0:
        raise ValueError('Lists must be longer than 0')

    while True:

        # Buffers for a batch of data
        x = np.zeros((batch_size, crop_size, crop_size, 1))
        y = np.zeros((batch_size, crop_size, crop_size, 3))

        # Get one image at a time
        for i in range(batch_size):

            random_ix = np.random.randint(low=0, high=len(x_list))

            x_curr = skimage.io.imread(x_list[random_ix])
            x_curr = normalize_image(x_curr)
            y_curr = skimage.io.imread(y_list[random_ix])

            crop_x, crop_y = random_cropping(x_curr, y_curr, crop_size)

            borders = True
            if borders:
                crop_y = add_borders(crop_y)
                crop_y = tf.keras.utils.to_categorical(crop_y)

            if augment:
                crop_x, crop_y = add_augmentation(crop_x, crop_y)

            # Save image to buffer
            x[i, :, :, 0] = crop_x
            y[i, :, :, 0:3] = crop_y

        # Return the buffer
        yield(x, y)


def main():
    ROOT = '../../data/ptrain_val/'
    MODEL_NAME = f"./models/{datetime.date.today().strftime('%Y%m%d')}_model"
    IMG_SIZE = 256

    # Import paths
    x_list = sorted(glob.glob(f'{ROOT}images/*.tif'))
    y_list = sorted(glob.glob(f'{ROOT}masks/*.tif'))
    log.info(f'x_list - {len(x_list)}, y_list - {len(y_list)}')

    # Train / valid split
    x_train, x_valid, y_train, y_valid = train_valid_split(x_list=x_list, y_list=y_list, valid_split=0.2)
    log.info(f'x_train - {len(x_train)}, x_valid - {len(x_valid)}')
    log.info(f'x_train - {x_train}')
    log.info(f'x_valid - {x_valid}')

    # Build model
    model = standard_unet()
    log.info('Model built.')

    # Compile model
    loss = tf.keras.losses.categorical_crossentropy
    metrics = [tf.keras.metrics.categorical_accuracy]
    optimizer = tf.keras.optimizers.Adam(lr=0.0001)
    model.compile(loss=loss, metrics=metrics, optimizer=optimizer)
    log.info('Model compiled.')

    # Callbacks
    callbacks = [tf.keras.callbacks.ModelCheckpoint(f'{MODEL_NAME}.h5', save_best_only=True),
                 tf.keras.callbacks.CSVLogger(filename=f'{MODEL_NAME}.csv'),
                 tf.keras.callbacks.TensorBoard(MODEL_NAME)]
    log.info(f'Callbacks for model "{MODEL_NAME}" generated.')

    # Build generators
    train_gen = random_sample_generator(
        x_list=x_train,
        y_list=y_train,
        augment=True,
        batch_size=16,
        bit_depth=16,
        crop_size=IMG_SIZE)
    val_gen = random_sample_generator(
        x_valid,
        y_valid,
        augment=False,
        batch_size=16,
        bit_depth=16,
        crop_size=IMG_SIZE)

    # Training
    log.info('Training starting.')
    # TODO use history to append to global comparison file
    history = model.fit_generator(generator=train_gen,
                                  steps_per_epoch=20,
                                  epochs=250,
                                  validation_data=val_gen,
                                  validation_steps=20,
                                  callbacks=callbacks,
                                  verbose=2)
    log.info('Training finished sucessfuly.')


if __name__ == "__main__":
    main()
