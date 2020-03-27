"""
Use the fluffy interface or import functions only.
"""

import numpy as np
import scipy.ndimage as ndi
import skimage.io
import tensorflow as tf


def predict_baseline(image, model, bit_depth=16):
    """
    Returns a binary or categorical model based prediction of an image.

    Args:
        - image (np.ndarray): Image to be predicted.
        - model (tf.keras.models.Model): Model used to predict the image.
        - bit_depth (int): Bit depth to normalize images. Model dependent.
    Returns:
        - pred (np.ndarray): Predicted image containing a probablilty map.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(
            f'image must be of type np.ndarray but is {type(image)}.')
    if not isinstance(model, tf.keras.models.Model):
        raise TypeError(
            f'model must be of type np.ndarray but is {type(model)}.')
    if not isinstance(bit_depth, int):
        raise TypeError(
            f'bit_depth must be of type np.ndarray but is {type(bit_depth)}.')
    if bit_depth == 0:
        raise ZeroDivisionError(f'bit_depth must not be zero')

    def __next_power(x, k=2):
        """ Calculates x's next higher power of k. """
        y, power = 0, 1
        while y < x:
            y = k**power
            power += 1
        return y

    model_output = model.layers[-1].output_shape[-1]
    if model_output not in [1, 3]:
        raise ValueError(f"model_output must be 1 or 3 but is {model_output}")

    pred = image * (1.0 / (2**bit_depth - 1))
    pad_bottom = __next_power(pred.shape[0]) - pred.shape[0]
    pad_right = __next_power(pred.shape[1]) - pred.shape[1]
    pred = np.pad(pred, ((0, pad_bottom), (0, pad_right)), "reflect")
    pred = model.predict(pred[None, ..., None]).squeeze()
    pred = pred[:pred.shape[0] - pad_bottom, :pred.shape[1] - pad_right]
    return pred


def add_instances(pred_mask):
    """
    Adds instances to a categorical prediction with three layers (background, foreground, borders).

    Args:
        - pred_mask (np.ndarray): Prediction mask to for instances should be added.
    Returns:
        - pred (np.ndarray): 2D mask containing unique values for every instance.
    """
    if not isinstance(pred_mask, np.ndarray):
        raise TypeError(
            f"pred_mask must be np.ndarray but is {type(pred_mask)}")
    if not pred_mask.ndim == 3:
        raise ValueError(
            f"pred_mask must have 3 dimensions but has {pred_mask.ndim}")

    foreground_eroded = ndi.binary_erosion(pred_mask[..., 1] > 0.5,
                                           iterations=2)
    markers = skimage.measure.label(foreground_eroded)
    background = 1 - pred_mask[..., 0] > 0.5
    foreground = 1 - pred_mask[..., 1] > 0.5
    watershed = skimage.morphology.watershed(foreground,
                                             markers=markers,
                                             mask=background)

    mask_new = []
    for i in np.unique(watershed):
        mask_curr = watershed == i
        mask_curr = ndi.binary_erosion(mask_curr, iterations=2)
        mask_new.append(mask_curr * i)
    return np.argmax(mask_new, axis=0)
