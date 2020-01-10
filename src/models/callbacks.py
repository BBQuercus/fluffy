import tensorflow as tf

def defaults(name_model, name_tb):
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(name_model, save_best_only=True),
        tf.keras.callbacks.CSVLogger(filename=f'{name}.csv'),
        tf.keras.callbacks.TensorBoard(name_tb)
    ]
    return callbacks