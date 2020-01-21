import numpy as np
import skimage


def _augment_scale(input_image, input_mask):
    '''
    Augments an image and mask by scaling a random amount.
    '''
    rand_scale = np.random.uniform(low=1, high=2)
    multichannel = True if input_mask.ndim == 3 else False

    output_image = skimage.transform.rescale(input_image, rand_scale, multichannel=False)
    output_mask = skimage.transform.rescale(input_mask, rand_scale, multichannel=multichannel)

    output_image = output_image[:input_image.shape[0], :input_image.shape[1]]
    output_mask = output_mask[:input_mask.shape[0], :input_mask.shape[1]]

    return output_image, output_mask


def _augment_flip(input_image, input_mask):
    '''
    Augments an image and mask by randomly flipping
    vertically and/or horizontally.
    '''

    rand_ud = np.random.randint(low=0, high=2)
    rand_lr = np.random.randint(low=0, high=2)

    if rand_ud == 1:
        input_image = np.flipud(input_image)
        input_mask = np.flipud(input_mask)

    if rand_lr == 1:
        input_image = np.fliplr(input_image)
        input_mask = np.fliplr(input_mask)

    return input_image, input_mask


def _augment_rotate(input_image, input_mask):
    '''
    Augments an image by randomly rotating 90 deg.
    '''
    rand_rot = np.random.randint(low=0, high=4)

    for _ in range(rand_rot):
        input_image = np.rot90(input_image)
        input_mask = np.rot90(input_mask)

    return input_image, input_mask


def _augment_swirl(input_image, input_mask):
    '''
    Augments an image and mask by applying small warps to
    immitate minor image deformations.
    '''
    rand_swirl = np.random.randint(low=0, high=4)
    multichannel = True if input_mask.ndim == 3 else False

    input_image = skimage.transform.swirl(input_image, strength=rand_swirl, radius=min(input_image.shape))

    if multichannel:
        for i in range(input_mask.shape[-1]):
            input_mask[:, :, i] = skimage.transform.swirl(input_mask[:, :, i], strength=rand_swirl, radius=min(input_image.shape))
    else:
        input_mask = skimage.transform.swirl(input_mask, strength=rand_swirl, radius=min(input_image.shape))

    return input_image, input_mask


def _augment_brightness(input_image):
    '''
    Augments an image by multiplying the pixel values
    changing the overall brightness.
    '''
    rand_bright = 1 + np.random.uniform(-0.75, 0.75)
    input_image *= rand_bright

    return input_image


def _augment_contrast(input_image):
    '''
    Augments image by randomly changing contrast in one of three ways.
    Contrast stretching, standard histogram equalization or adaptive
    equalization.
    '''
    rand_total = np.random.randint(low=0, high=1)
    rand = np.random.randint(low=0, high=1)

    # Contrast stretching
    if (rand_total == 0 and rand == 0):
        p2, p98 = np.percentile(input_image, (2, 98))
        input_image = skimage.exposure.rescale_intensity(input_image, in_range=(p2, p98))

    # Adaptive Equalization
    if (rand_total == 0 and rand == 1):
        input_image = skimage.exposure.equalize_adapthist(input_image, clip_limit=0.03)

    return input_image


def _convert_to_binary(input_mask):
    ''' Converts multi-dimensional, non-binary mask to binary.'''
    for i in range(input_mask.shape[-1]):
        input_mask[:, :, i] = input_mask[:, :, i] > 0.5
    return input_mask


def default(input_image, input_mask):
    '''
    '''

    input_image, input_mask = _augment_scale(input_image, input_mask)

    input_image, input_mask = _augment_flip(input_image, input_mask)

    input_image, input_mask = _augment_rotate(input_image, input_mask)

    input_image, input_mask = _augment_swirl(input_image, input_mask)

    input_image = _augment_brightness(input_image)

    input_image = _augment_contrast(input_image)

    if input_mask.ndim == 3:
        input_mask = _convert_to_binary(input_mask)

    return input_image, input_mask
