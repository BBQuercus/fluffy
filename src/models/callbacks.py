import tensorflow as tf
from tensorflow import keras


def defaults(path_model, path_csv, path_tb):
    '''
    Returns a list of callbacks used by default.

    Args:
        path_model (dir): Filename where model should be saved.
        path_csv (dir): Filename where csv log should be saved.
        path_tb (dir): Tensorboard log directory.

    Returns:
        list: default callbacks.
    '''
    # TODO assert names are properly

    return [
        # keras.callbacks.LearningRateScheduler(scheduler)
        # keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
        keras.callbacks.ModelCheckpoint(path_model, save_best_only=True),
        keras.callbacks.CSVLogger(path_csv),
        keras.callbacks.TensorBoard(path_tb),
        keras.callbacks.EarlyStopping(patience=5)
    ]


# def scheduler(epoch):
#     if epoch < 10:
#         return 0.001
#     else:
#         return 0.001 * tf.math.exp(0.1 * (10 - epoch))
