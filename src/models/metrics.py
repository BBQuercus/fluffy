import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

'''
Many metrics below were adapted from Dominik Müller,
his repo can be found @frankkramer-lab/MIScnn.
'''


def weighted_crossentropy(y_true, y_pred, weights=[1., 1., 10.]):
    '''
    Weighted version of tf.keras.objectives.categorical_crossentropy
    '''
    unweighted_losses = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=y_pred)

    class_weights = tf.constant([[[weights]]])
    weights = tf.reduce_sum(class_weights*y_true, axis=-1)

    weighted_losses = weights*unweighted_losses
    loss = tf.reduce_mean(weighted_losses)
    return loss


def identify_axis(shape):
    if len(shape) == 5:
        return [1, 2, 3]
    elif len(shape) == 4:
        return [1, 2]
    else:
        raise ValueError('Metric: Shape of tensor is neither 2D or 3D.')


def dice_weighted(weights):
    weights = K.variable(weights)

    def weighted_loss(y_true, y_pred, ep=1e-08):
        axis = identify_axis(y_true.get_shape())
        intersection = y_true * y_pred
        intersection = K.sum(intersection, axis=axis)
        y_true = K.sum(y_true, axis=axis)
        y_pred = K.sum(y_pred, axis=axis)
        dice = ((2 * intersection) + ep) / (y_true + y_pred + ep)
        dice = dice * weights
        return -dice
    return weighted_loss


def dice_coefficient(y_true, y_pred, ep=1e-08):
    '''
    '''
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + ep) / (K.sum(y_true_f) + K.sum(y_pred_f) + ep)
    return dice


def dice_coefficient_loss(y_true, y_pred):
    return 1-dice_coefficient(y_true, y_pred)


def dice_soft(y_true, y_pred, ep=1e-08):
    axis = identify_axis(y_true.get_shape())

    intersection = y_true * y_pred
    intersection = K.sum(intersection, axis=axis)
    y_true = K.sum(y_true, axis=axis)
    y_pred = K.sum(y_pred, axis=axis)

    dice = ((2 * intersection) + ep) / (y_true + y_pred + ep)
    dice = K.mean(dice)
    return dice


def dice_soft_loss(y_true, y_pred):
    return 1-dice_soft(y_true, y_pred)


def dice_crossentropy(y_true, y_pred):
    dice = dice_soft_loss(y_true, y_pred)
    crossentropy = K.categorical_crossentropy(y_true, y_pred)
    crossentropy = K.mean(crossentropy)
    return dice + crossentropy


def tversky_loss(y_true, y_pred, ep=1e-08):
    '''
    Sadegh et al. 2017
        - alpha=beta=0.5 : dice coefficient
        - alpha=beta=1   : jaccard
        - alpha+beta=1   : produces set of F*-scores
    '''
    alpha = 0.5
    beta = 0.5
    # Calculate Tversky per class
    axis = identify_axis(y_true.get_shape())
    tp = K.sum(y_true * y_pred, axis=axis)
    fn = K.sum(y_true * (1-y_pred), axis=axis)
    fp = K.sum((1-y_true) * y_pred, axis=axis)
    tversky_class = (tp + ep)/(tp + alpha*fn + beta*fp + ep)
    tversky = K.sum(tversky_class, axis=[-1])
    # Find number of classes
    n = K.cast(K.shape(y_true)[-1], 'float32')
    return n-tversky


def tversky_crossentropy(y_truth, y_pred):
    tversky = tversky_loss(y_truth, y_pred)
    crossentropy = K.categorical_crossentropy(y_truth, y_pred)
    crossentropy = K.mean(crossentropy)
    return tversky + crossentropy


def active_contour_loss(y_true, y_pred):
    '''
    Active contour loss from the arxiv paper "Learning Active Contour Models
    for Medical Image Segmentation" by Chen, Xu, et al.
    Awesome simple github repo at @xuuuuuuchen/Active-Contour-Loss
    '''
    # Length term – horizontal / vertical directions
    x = y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :]
    y = y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1]

    delta_x = x[:, :, 1:, :-2]**2
    delta_y = y[:, :, :-2, 1:]**2
    delta_u = K.abs(delta_x + delta_y)

    length = 1 * K.sum(K.sqrt(delta_u + 1e-08))  # equ.(11)

    # Region term
    foreground = np.ones((256, 256))
    background = np.zeros((256, 256))

    region_in = K.abs(K.sum(y_pred[:, 0, :, :] * ((y_true[:, 0, :, :]-foreground)**2)))  # equ.(12)
    region_out = K.abs(K.sum((1-y_pred[:, 0, :, :]) * ((y_true[:, 0, :, :]-background)**2)))  # equ.(12)
    lambdaP = 1

    loss = length + lambdaP * (region_in + region_out)

    return loss


def defaults():
    '''
    '''
    return [
        # TODO fix magic number
        keras.metrics.MeanIoU(num_classes=3),
        keras.metrics.AUC()
    ]
