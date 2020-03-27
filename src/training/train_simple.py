"""
train_simple.py

Trains a simple model with default parameters.

No cross validation / hyperparameter optimization is performed.
This script allows for quick testing to see if a model has potential for a longer
    run using the main "train_model.py" script.
"""

import click
import datetime
import glob
import logging
import tensorflow as tf

import models
import dirtools
import data

LOG_FORMAT = "%(levelname)s %(asctime)s - %(filename)s %(funcName)s %(lineno)s - %(message)s"
logging.basicConfig(filename="./train_model.log",
                    level=logging.DEBUG,
                    format=LOG_FORMAT,
                    filemode="a")
log = logging.getLogger()


@click.command()
@click.option("--model_type", default="categorical", help="If model is binary or categorical.")
@click.option("--name", default="model", help="Name of model.")
def main(model_type, name):
    if not all(isinstance(i, str) for i in [model_type, name]):
        return TypeError(f"model_type, name must be str but are {type(model_type)}, {type(name)}")
    if model_type not in ["binary", "categorical"]:
        raise ValueError(f"model_type must be binary or categorical but is {model_type}.")

    ROOT = "../../data/processed/train_val/"
    MODEL_NAME = f"./models/{datetime.date.today().strftime('%Y%m%d')}_{name}"
    BINARY = True if model_type == "binary" else False
    IMG_SIZE = 256
    log.info(f"Model_type is {model_type}")

    # Import paths
    x_list = sorted(glob.glob(f"{ROOT}images/*.tif"))
    y_list = sorted(glob.glob(f"{ROOT}masks/*.tif"))
    log.info(f"x_list - {len(x_list)}, y_list - {len(y_list)}")

    # Train / valid split
    x_train, x_valid, y_train, y_valid = dirtools.train_valid_split(x_list=x_list, y_list=y_list, valid_split=0.2)
    log.info(f"x_train - {len(x_train)}, x_valid - {len(x_valid)}")
    log.info(f"x_train - {x_train}")
    log.info(f"x_valid - {x_valid}")

    # Build model
    model = models.unet(binary=BINARY)
    log.info("Model built.")

    # Compile model
    if BINARY:
        loss = tf.keras.losses.binary_crossentropy
        metrics = [tf.keras.metrics.binary_accuracy]
    else:
        loss = tf.keras.losses.categorical_crossentropy
        metrics = [tf.keras.metrics.categorical_accuracy]
    optimizer = tf.keras.optimizers.Adam(lr=0.0001)
    model.compile(loss=loss, metrics=metrics, optimizer=optimizer)
    log.info("Model compiled.")

    # Callbacks
    callbacks = [tf.keras.callbacks.ModelCheckpoint(f"{MODEL_NAME}.h5", save_best_only=True),
                 tf.keras.callbacks.CSVLogger(filename=f"{MODEL_NAME}.csv"),
                 tf.keras.callbacks.TensorBoard(MODEL_NAME)]
    log.info(f"Callbacks for model '{MODEL_NAME}' generated.")

    # Build generators
    train_gen = data.sample_generator(
        x_list=x_train,
        y_list=y_train,
        binary=BINARY,
        augment=True,
        batch_size=16,
        bit_depth=16,
        crop_size=IMG_SIZE)
    val_gen = data.sample_generator(
        x_list=x_valid,
        y_list=y_valid,
        binary=BINARY,
        augment=False,
        batch_size=16,
        bit_depth=16,
        crop_size=IMG_SIZE)

    # Training
    log.info("Training starting.")
    # TODO use history to append to global comparison file
    history = model.fit_generator(generator=train_gen,
                                  steps_per_epoch=20,
                                  epochs=250,
                                  validation_data=val_gen,
                                  validation_steps=20,
                                  callbacks=callbacks,
                                  verbose=2)
    log.info("Training finished sucessfully :).")


if __name__ == "__main__":
    main()
