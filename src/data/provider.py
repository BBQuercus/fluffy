import numpy as np
import scipy.ndimage as ndi
from tensorflow import keras

import data.augment


def _add_borders(input_mask, size):
    '''
    '''

    mask = input_mask > 0
    for i in np.unique(input_mask):
        curr_mask = np.where(input_mask == i, 1, 0)
        mask_dilated = ndi.binary_dilation(curr_mask, iterations=size)
        mask_eroded = ndi.binary_erosion(curr_mask, iterations=size)
        border = np.logical_xor(mask_dilated, mask_eroded)
        mask = np.where(border, 2, mask)

    return mask


def _normalize_image(input_image, bit_depth):
    return input_image.astype(np.float32) / ((2**bit_depth)-1.)


def _grayscale_to_rgb(input_image):
    return np.stack((input_image,)*3, axis=-1)


def load_train_seg(input_image, input_mask, **kwargs):
    '''
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
    one = np.random.randint(low=0, high=input_image.shape[0]-img_size) if input_image.shape[0]>img_size else 0
    two = np.random.randint(low=0, high=input_image.shape[1]-img_size) if input_image.shape[1]>img_size else 0
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
