import numpy as np
import scipy.ndimage as ndi
from tensorflow import keras

import data.augment


def _add_borders(input_mask, size, touching_only=False):
    '''
    Returns a mask containing the borders of all objects.

    Args:
        input_mask (np.array): Mask to be processed.
        size (int): Size of borders (iterations of dilution / erosion).
        touching_only (bool): If true returns only overlapping borders
            after dilated. If false returns all borders.

    Returns:
        mask (np.array): Mask containing borders.
    '''

    if size < 1:
        return input_mask

    mask = input_mask > 0
    borders = []
    for i in np.unique(input_mask):
        curr_mask = np.where(input_mask == i, 1, 0)
        mask_dilated = ndi.binary_dilation(curr_mask, iterations=size)
        mask_eroded = ndi.binary_erosion(curr_mask, iterations=size)
        curr_border = np.logical_xor(mask_dilated, mask_eroded)
        borders.append(curr_border)

    mask = np.sum(borders[1:], axis=0)

    if touching_only:
        return mask > 1
    else:
        return mask > 0


def _normalize_image(input_image, bit_depth):
    ''' Normalizes image based on bit depth. '''
    return input_image.astype(np.float32) / ((2**bit_depth)-1.)


def _grayscale_to_rgb(input_image):
    ''' Converts grayscale image to RGB. '''
    assert input_image.ndim != 2, 'Image not grayscale.'
    return np.stack((input_image,)*3, axis=-1)


def load_train_seg(input_image, input_mask, **kwargs):
    '''
    Preprocessing of image and mask for training. Main difference to
    load_valid_seg is ability of data augmentation.

    Args:
        input_image (np.array): Image to be processed.
        input_mask (np.array): Corresponding mask.
        ––– **kwargs
        img_size (int): Desired size of image.
        bit_depth (int): Bit depth of image.
        border_size (int): Size of border dilutions,
            set to zero if no borders are desired.
        convert_to_rgb (bool): If True, images will be converted to RGB.
        ––– **kwargs +
        Additional kwargs used for data augmentation are
            explained in the function data.augment.default().

    Returns:
        input_image (np.array):
        input_mask (np.array):
    '''
    img_size = kwargs.get('img_size', 224)
    bit_depth = kwargs.get('bit_depth', 16)
    border_size = kwargs.get('border_size', 2)
    convert_to_rgb = kwargs.get('convert_to_rgb', True)

    assert type(input_image) == np.ndarray and type(input_mask) == np.ndarray
    # assert input_image.shape[:2] == input_mask.shape[:2]
    assert input_image.shape[0] >= img_size
    assert input_image.shape[1] >= img_size

    # Cropping
    one = np.random.randint(low=0, high=input_image.shape[0]-img_size) if input_image.shape[0] > img_size else 0
    two = np.random.randint(low=0, high=input_image.shape[1]-img_size) if input_image.shape[1] > img_size else 0
    input_image = input_image[one:one+img_size, two:two+img_size]
    input_mask = input_mask[one:one+img_size, two:two+img_size]

    # Augmentation
    # input_image, input_mask = data.augment

    # Normalization
    input_image = _normalize_image(input_image, bit_depth)
    if convert_to_rgb:
        input_image = _grayscale_to_rgb(input_image)

    # Add border to mask
    input_mask = _add_borders(input_mask, border_size)
    input_mask = keras.utils.to_categorical(input_mask)

    return input_image, input_mask


def train_generator_seg(npy_images, npy_masks, **kwargs):
    '''
    Generator yielding batches of training images and masks.

    Args:
        npy_images (np.array): Array containing all images.
            Images are checked and preprocessed.
        npy_masks (np.array): Array containing all masks.
        ––– **kwargs
        batch_size (int): Size of minibatch yielded by generator.
        ––– **kwargs +
        Additional kwargs are used in the function load_train_seg.

    Returns:
        np.array (generator): Preprocessed images and masks
            for training.
    '''
    assert len(npy_images) != 0
    assert len(npy_masks) == len(npy_masks)
    for img in npy_images:
        assert img.ndim == 2, 'Images not grayscale'

    batch_size = kwargs.get('batch_size', 16)

    while True:
        images = []
        masks = []
        for _ in range(batch_size):
            ix = np.random.randint(len(npy_images))
            image, mask = load_train_seg(npy_images[ix], npy_masks[ix], **kwargs)
            images.append(image)
            masks.append(mask)

    yield np.array(images), np.array(masks)


def load_val_seg(input_image, input_mask, **kwargs):
    '''
    Preprocessing of image and mask for validation. Main difference to
    load_train_seg is lack of data augmentation.

    Args:
        input_image (np.array): Image to be processed.
        input_mask (np.array): Corresponding mask.
        ––– **kwargs
        img_size (int): Desired size of image.
        bit_depth (int): Bit depth of image.
        border_size (int): Size of border dilutions,
            set to zero if no borders are desired.
        convert_to_rgb (bool): If True, images will be converted to RGB.

    Returns:
        input_image (np.array):
        input_mask (np.array):
    '''
    img_size = kwargs.get('img_size', 224)
    bit_depth = kwargs.get('bit_depth', 16)
    border_size = kwargs.get('border_size', 2)
    convert_to_rgb = kwargs.get('convert_to_rgb', True)

    assert type(input_image) == np.ndarray and type(input_mask) == np.ndarray
    assert input_image.shape[:2] == input_mask.shape[:2]
    assert input_image.shape[0] >= img_size
    assert input_image.shape[1] >= img_size

    # Cropping
    one = np.random.randint(low=0, high=input_image.shape[0]-img_size) if input_image.shape[0] > img_size else 0
    two = np.random.randint(low=0, high=input_image.shape[1]-img_size) if input_image.shape[1] > img_size else 0
    input_image = input_image[one:one+img_size, two:two+img_size]
    input_mask = input_mask[one:one+img_size, two:two+img_size]

    # Normalization
    input_image = _normalize_image(input_image, bit_depth)
    if convert_to_rgb:
        input_image = _grayscale_to_rgb(input_image)

    # Add border to mask
    input_mask = _add_borders(input_mask, border_size)
    input_mask = keras.utils.to_categorical(input_mask)

    return input_image, input_mask


def val_generator_seg(npy_images, npy_masks, **kwargs):
    '''
    Generator yielding batches of validation or inference images and masks.

    Args:
        npy_images (np.array): Array containing all images.
            Images are checked and preprocessed.
        npy_masks (np.array): Array containing all masks.
        ––– **kwargs
        batch_size (int): Size of minibatch yielded by generator.
        ––– **kwargs +
        Additional kwargs are used in the function load_valid_seg.

    Returns:
        np.array (generator): Preprocessed images and masks
            for validation.
    '''
    assert len(npy_images) != 0
    assert len(npy_masks) == len(npy_masks)
    for img in npy_images:
        assert img.ndim == 2, 'Images not grayscale'

    batch_size = kwargs.get('batch_size', 16)

    while True:
        images = []
        masks = []
        for _ in range(batch_size):
            ix = np.random.randint(len(npy_images))
            image, mask = load_val_seg(npy_images[ix], npy_masks[ix], **kwargs)
            images.append(image)
            masks.append(mask)

    yield np.array(images), np.array(masks)
