import tensorflow as tf

import models.loss

defaults = {
    # Architecture related
    'input_size' : 256,
    'model_depth' : 3,
    'model_width' : 16,
    'bn_momentum' : 0.9,
    'conv_activation' : tf.keras.activations.relu,

    # Compilation related
    'learning_rate' : 0.01,
    'optimizer' : tf.keras.optimizers.Adam,
    'loss' : models.loss.dice_coef_binary_loss,

    # Training related
    'batch_size' : 32,
    'epochs' : 100,

    # Data-preprocessing
    'scaling' : True,
    'cropping' : True,
    'flipping' : True,
    'padding' : True,
    'rotation' : True,
    'brightness' : True,
    'contrast' : True,
}
