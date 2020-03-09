import logging
import numpy as np
import scipy.ndimage as ndi
import skimage.io
import tensorflow as tf

LOG_FORMAT = '%(levelname)s %(asctime)s - %(filename)s %(funcName)s %(lineno)s - %(message)s'
logging.basicConfig(filename='./train_model.log',
                    level=logging.DEBUG,
                    format=LOG_FORMAT,
                    filemode='a')
log = logging.getLogger()


def add_borders(mask, border_size=2, to_categorical=True):
    '''
    Adds borders to labeled masks.

    Args:
        - mask (np.ndarray): Mask with uniquely labeled
            object to which borders will be added.
        - border_size (int): Size of border in pixels.
        - to_categorical (bool): If true, will be converted into image with 
            three layers. One per category.
    Returns:
        - output_mask (np.ndarray): Mask with three channels –
            Background (0), Objects (1), and Borders (2).
    '''
    if not isinstance(mask, np.ndarray):
        raise TypeError(f'input_mask must be a np.ndarray but is a {type(mask)}.')
    if not isinstance(border_size, int):
        raise TypeError(f'size must be an int but is a {type(border_size)}.')

    borders = np.zeros(mask.shape)
    for i in np.unique(mask):
        curr_mask = np.where(mask == i, 1, 0)
        mask_dil = ndi.morphology.binary_dilation(curr_mask, iterations=border_size)
        mask_ero = ndi.morphology.binary_erosion(curr_mask, iterations=border_size)
        mask_border = np.logical_xor(mask_dil, mask_ero)
        borders[mask_border] = i
    output_mask = np.where(borders > 0, 2, mask > 0)

    if to_categorical:
        return tf.keras.utils.to_categorical(output_mask)
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
        raise TypeError(f'image, mask must be a np.ndarray but are {type(image)}, {type(mask)}.')

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
        raise TypeError(f'image, mask must be np.ndarray but is {type(image)}, {type(mask)}.')
    if not isinstance(crop_size, int):
        raise TypeError(f'crop_size must be an int but is {type(crop_size)}.')
    if not image.shape[:2] == mask.shape[:2]:
        raise ValueError(f'image, mask must be of same shape: {image.shape[:2]} != {mask.shape[:2]}.')
    if crop_size == 0:
        raise ValueError('crop_size must be larger than 0.')

    start_dim1 = np.random.randint(low=0, high=image.shape[0] - crop_size) if image.shape[0] > crop_size else 0
    start_dim2 = np.random.randint(low=0, high=image.shape[1] - crop_size) if image.shape[1] > crop_size else 0

    crop_image = image[start_dim1:start_dim1 + crop_size, start_dim2:start_dim2 + crop_size]
    crop_mask = mask[start_dim1:start_dim1 + crop_size, start_dim2:start_dim2 + crop_size]

    return crop_image, crop_mask


def normalize_image(image, bit_depth=16):
    ''' Normalizes image by bit_depth. '''
    if not isinstance(image, np.ndarray):
        raise TypeError(f'image must be np.ndarray but is {type(image)}.')
    if not isinstance(bit_depth, int):
        raise TypeError(f'bit_depth must be int but is {type(int)}.')
    if bit_depth == 0:
        raise ZeroDivisionError('bit_depth must not be zero.')

    return image * (1./(2**bit_depth - 1))


# TODO refactor to simpler function
def random_sample_generator(
    x_list, y_list,
    binary=True,
    augment=True,
    batch_size=16,
    bit_depth=16,
    crop_size=256
        ):
    '''
    Yields a generator for training examples with augmentation.

    Args:
        - x_list (list): List containing filepaths to all usable images.
        - y_list (list): List containing filepaths to all usable masks.
        - binary (bool): If batches are for binary or categorical models.
        - augment (bool): If augmentation should be done.
        - batch_size (int): Size of one mini-batch.
        - bit_depth (int): Bit depth of images to normalize.
        - crop_size (int): Size to crop images to.
    Returns:
        - Generator: List of augmented training examples of
            size (batch_size, img_size, img_size, 3)
    '''
    if not all(isinstance(i, list) for i in [x_list, y_list]):
        raise TypeError(f'x_list, y_list must be list but are {type(x_list)}, {type(y_list)}.')
    if not all(isinstance(y, str) for y in y_list):
        raise TypeError('Elements of y_list must be a str.')
    if not all(isinstance(i, bool) for i in [binary, augment]):
        raise TypeError(f'binary, augment must be bool but is {type(binary)}, {type(augment)}.')
    if not all(isinstance(i, int) for i in [batch_size, bit_depth, crop_size]):
        raise TypeError(f'batch_size, bit_depth, crop_size must be int but are {type(batch_size)}, {type(bit_depth)}, {type(crop_size)}.')
    if len(x_list) != len(y_list):
        raise ValueError(f'Lists must be of equal length: {len(x_list)} != {len(y_list)}.')
    if len(x_list) < 0:
        raise ValueError('Lists must be longer than 0.')

    channels = 1 if binary else 3

    while True:

        # Buffers for a batch of data
        x = np.zeros((batch_size, crop_size, crop_size, 1))
        y = np.zeros((batch_size, crop_size, crop_size, channels))

        # Get one image at a time
        for i in range(batch_size):

            random_ix = np.random.randint(low=0, high=len(x_list))

            x_curr = skimage.io.imread(x_list[random_ix])
            x_curr = normalize_image(x_curr)
            y_curr = skimage.io.imread(y_list[random_ix])

            crop_x, crop_y = random_cropping(x_curr, y_curr, crop_size)

            if not binary:
                crop_y = add_borders(crop_y)

            if augment:
                crop_x, crop_y = add_augmentation(crop_x, crop_y)

            # Save image to buffer
            x[i, :, :, 0] = crop_x
            if binary:
                y[i, :, :, 0] = crop_y
            else:
                y[i, :, :, :channels] = crop_y

        # Return the buffer
        yield(x, y)
