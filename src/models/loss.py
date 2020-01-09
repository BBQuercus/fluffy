import tensorflow as tf
import tensorflow.keras.backend as K

'''
- 
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

def dice_coef_binary(y_true, y_pred, smooth=1e-7):
    '''
    Dice coefficient for 2 categories. Ignores background pixel label 0
    Pass to model as metric during compile statement
    '''
    y_true_f = K.flatten(K.one_hot(K.cast(y_true, 'int32'), num_classes=2)[...,1:])
    y_pred_f = K.flatten(y_pred[...,1:])
    intersect = K.sum(y_true_f * y_pred_f, axis=-1)
    denom = K.sum(y_true_f + y_pred_f, axis=-1)
    return K.mean((2. * intersect / (denom + smooth)))

def dice_coef_binary_loss(y_true, y_pred):
    '''
    Dice loss to minimize. Pass to model as loss during compile statement
    '''
    return 1 - dice_coef_binary(y_true, y_pred)

def active_contour_loss(y_true, y_pred): 
    '''
    Active contour loss from the arxiv paper "Learning Active Contour Models
    for Medical Image Segmentation" by Chen, Xu, et al.
    Awesome simple github repo at @xuuuuuuchen/Active-Contour-Loss
    '''
    # Length term – horizontal / vertical directions
    x = y_pred[:,:,1:,:] - y_pred[:,:,:-1,:]
    y = y_pred[:,:,:,1:] - y_pred[:,:,:,:-1]

    delta_x = x[:,:,1:,:-2]**2
    delta_y = y[:,:,:-2,1:]**2
    delta_u = K.abs(delta_x + delta_y) 

    length = 1 * K.sum(K.sqrt(delta_u + 1e-08)) # equ.(11)

    # Region term
    foreground = np.ones((256, 256))
    background = np.zeros((256, 256))

    region_in = K.abs(K.sum(y_pred[:,0,:,:] * ((y_true[:,0,:,:]-foreground)**2))) # equ.(12)
    region_out = K.abs(K.sum((1-y_pred[:,0,:,:]) * ((y_true[:,0,:,:]-background)**2))) # equ.(12)
    lambdaP = 1

    loss =  length + lambdaP * (region_in + region_out) 

    return loss