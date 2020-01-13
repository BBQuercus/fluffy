import tensorflow as tf
from tensorflow import keras


def scheduler(epoch):
    if epoch < 10:
        return 0.001
    else:
        return 0.001 * tf.math.exp(0.1 * (10 - epoch))


def defaults(path_model, path_csv, path_tb):
    '''
    '''
    return [
        # keras.callbacks.LearningRateScheduler(scheduler)
        # keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
        keras.callbacks.ModelCheckpoint(path_model, save_best_only=True),
        keras.callbacks.CSVLogger(path_csv),
        keras.callbacks.TensorBoard(path_tb),
        keras.callbacks.EarlyStopping(patience=5)
    ]
