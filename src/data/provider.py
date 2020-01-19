import numpy as np
import scipy.ndimage as ndi

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

    mask = input_mask > 0

    borders = []
    for i in np.unique(input_mask):
        curr_mask = np.where(input_mask == i, 1, 0)
        mask_dilated = ndi.binary_dilation(curr_mask, iterations=size)
        mask_eroded = ndi.binary_erosion(curr_mask, iterations=size)
        curr_border = np.logical_xor(mask_dilated, mask_eroded)
        borders.append(curr_border)

    borders = np.sum(borders[1:], axis=0)
    cutoff = 1 if touching_only else 0

    mask_borders = (borders > cutoff).astype(np.float32)
    mask_foreground = (mask + (borders > cutoff)) - mask_borders
    mask_background = 1 - (mask + (borders > cutoff))

    mask = np.stack((mask_background, mask_foreground, mask_borders), axis=-1)

    return mask


def _normalize_image(input_image, bit_depth):
    ''' Normalizes image based on bit depth. '''
    return input_image.astype(np.float32) / ((2**bit_depth)-1.)


# def _grayscale_to_rgb(input_image):
#     ''' Converts grayscale image to RGB. '''
#     assert input_image.ndim != 2, 'Image not grayscale.'
#     return np.stack((input_image,)*3, axis=-1)


def load_seg(input_image, input_mask, training=False, **kwargs):
    '''
    Preprocessing of image and mask. Can be used for training or
    validation (without data augmentation).

    Args:
        input_image (np.array): Image to be processed.
        input_mask (np.array): Corresponding mask.
        --- **kwargs
        img_size (int): Desired size of image.
        bit_depth (int): Bit depth of image.
        border (bool): If True, will add borders
        border_size (int): Size of border dilutions,
            set to zero if no borders are desired.
        touching_only (bool): If True, only touching borders
            will be used.

    Returns:
        input_image (np.array):
        input_mask (np.array):
    '''
    img_size = kwargs.get('img_size', 256)
    bit_depth = kwargs.get('bit_depth', 16)
    border = kwargs.get('border', True)
    border_size = kwargs.get('border_size', 2)
    touching_only = kwargs.get('touching_only', False)

    assert type(input_image) == np.ndarray and type(input_mask) == np.ndarray
    assert input_image.shape[:2] == input_mask.shape[:2]
    assert input_image.shape[0] >= img_size
    assert input_image.shape[1] >= img_size

    # Normalization
    input_image = _normalize_image(input_image, bit_depth)

    # Add border to mask
    if border:
        input_mask = _add_borders(input_mask, border_size, touching_only)

    # Cropping
    one, two = 0, 0
    if input_image.shape[0] > img_size:
        one = np.random.randint(low=0, high=input_image.shape[0]-img_size)
    if input_image.shape[1] > img_size:
        two = np.random.randint(low=0, high=input_image.shape[1]-img_size)

    input_image = input_image[one:one+img_size, two:two+img_size]
    input_mask = input_mask[one:one+img_size, two:two+img_size]

    # Data augmentation for training
    if training:
        input_image, input_mask = data.augment.default(input_image, input_mask)

    return input_image, input_mask


def generator_seg(npy_images, npy_masks, training=False, **kwargs):
    '''
    Generator yielding batches  orvalidation inference images and masks.

    Args:
        npy_images (np.array): Array containing all images.
            Images are checked and preprocessed.
        npy_masks (np.array): Array containing all masks.
        training (bool): If True will prepare documents for training.
        --- **kwargs
        img_size (int): Size to which images get reshaped.
        batch_size (int): Size of minibatch yielded by generator.
        --- **kwargs +
        Additional kwargs are used in the function load_valid_seg.

    Returns:
        np.array (generator): Preprocessed images and masks
            for validation.
    '''
    assert len(npy_images) != 0
    assert len(npy_masks) == len(npy_masks)
    for img in npy_images:
        assert img.ndim == 2, 'Images not grayscale'
    for msk in npy_masks:
        assert msk.ndim == 2, 'Masks not binary'

    img_size = kwargs.get('img_size', 256)
    batch_size = kwargs.get('batch_size', 16)
    border = kwargs.get('border', True)
    channels = 3 if border else 1

    while True:
        images = np.zeros((batch_size, img_size, img_size, 1))
        masks = np.zeros((batch_size, img_size, img_size, channels))

        for i in range(batch_size):
            ix = np.random.randint(len(npy_images))
            image, mask = load_seg(npy_images[ix], npy_masks[ix], training, **kwargs)

            if mask.shape[2] == 2:
                import skimage.io
                skimage.io.imsave('~/Downloads/img0.png', mask[:,:,0])
                skimage.io.imsave('~/Downloads/img1.png', mask[:,:,1])

            images[i, :, :, 0] = image
            masks[i, :, :, :channels] = mask

        yield images, masks
