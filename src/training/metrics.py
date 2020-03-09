# import tensorflow as tf
import tensorflow.keras.backend as K


def dice_coef(y_true, y_pred, epsilon=1e-8):
    '''
    Calculates the dice coefficient on a pixel basis.
    Dice = (2TP) / (2TP + FP + FN).
    In comparison to IoU dice weights true positives higher by the 2x multiplication.
    '''
    y_true_f = K.batch_flatten(y_true)
    y_pred_f = K.batch_flatten(y_pred)
    intersection = 2. * K.sum(y_true_f * y_pred_f, axis=1, keepdims=True) + epsilon
    union = K.sum(y_true_f, axis=1, keepdims=True) + K.sum(y_pred_f, axis=1, keepdims=True) + epsilon
    return K.mean(intersection / union)
