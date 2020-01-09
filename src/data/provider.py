import numpy as np
import scipy.ndimage as ndi
import skimage

from tensorflow import keras

import data.augment

def _add_borders(input_mask):
    '''
    '''
    
    mask = input_mask>0

    for i in np.unique(input_mask):
        curr_mask = np.where(input_mask==i, 1, 0)
        mask_dilated = ndi.binary_dilation(curr_mask)
        mask_eroded = ndi.binary_erosion(curr_mask)
        border = np.logical_xor(mask_dilated, mask_eroded)
        mask = np.where(border, 2, mask)

    return mask

def _normalize_image(input_image, bit_depth):
    return input_image.astype(np.float32) / ((2**bit_depth)-1.)

def load_train_segmentation(input_image, input_mask, img_size, bit_depth):
    '''
    '''

    # input_image = skimage.io.imread(file_image)
    # input_mask = skimage.io.imread(file_mask)
    assert input_image == np.ndarray and input_mask == np.ndarray
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

    # Add border to mask
    input_mask = _add_borders(input_mask)
    input_mask = keras.utils.to_categorical(input_mask)
    
    return input_image, input_mask

def generator_train_segmentation(npy_images, npy_masks, img_size, bit_depth, batch_size):
    '''
    '''

    assert len(list_images)!=0
    assert len(list_images)==len(list_masks)
    assert list_images[0].shape[3]==1, 'Images not grayscale'
    assert list_masks[0].shape[3]==3, 'Masks not with one-hot borders'

    while True:
        images = np.zeros((batch_size, img_size, img_size, 1))
        masks = np.zeros((batch_size, img_size, img_size, 3))

        for i in range(batch_size): 
            ix = np.random.randint(len(list_images))
            image, mask = load_train_segmentation(list_images[ix], list_masks[ix], img_size, bit_depth)
            images[i,:,:,0] = image
            masks[i,:,:,:] = mask

    yield(images, masks)