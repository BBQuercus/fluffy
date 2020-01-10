import tensorflow as tf

import models.loss
import models.callbacks

defaults = {
    # Architecture related
    'img_size' : 256,
    'depth' : 3,
    'width' : 16,
    'momentum' : 0.9,
    'activation' : tf.keras.activations.relu,
    'categories' : 3,

    # Compilation related
    'lr' : 0.01,
    'optimizer' : tf.keras.optimizers.Adam,
    'loss' : models.loss.dice_coef_binary_loss,
    # 'metrics' : models.metrics.defaults,
    'callbacks' : models.callbacks.defaults,

    # Training related
    'epochs' : 100,
    'validation_freq' : 2,
    'verbose' : 2,

    # Data-preprocessing
    'batch_size' : 32,
    'scaling' : True,
    'cropping' : True,
    'flipping' : True,
    'padding' : True,
    'rotation' : True,
    'brightness' : True,
    'contrast' : True,
}
