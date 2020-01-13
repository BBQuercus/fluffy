import click
import os
import shutil
import numpy as np
import tensorflow as tf
from tensorflow import keras

import data
import yolo


@click.command()
@click.option('--bboxes', help='Location of txt file containing bounding boxes')
def main(bboxes):
    trainset = data.Dataset(bboxes)
    logdir = "./data/log"
    steps_per_epoch = len(trainset)
    global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)
    warmup_steps = 2 * steps_per_epoch  # WARMUP_EPOCHS
    epochs = 30
    total_steps = epochs * steps_per_epoch  # EPOCHS

    ANCHORS = '/Users/beichenberger/Github/fluffy-guide/src/yolov3/anchors/anchors.txt'
    ANCHORS = yolo.get_anchors(ANCHORS)

    input_layer = keras.layers.Input([416, 416, 3])
    model = yolo.model_obj(input_layer, ANCHORS)
    model.summary()

    optimizer = tf.keras.optimizers.Adam()
    if os.path.exists(logdir):
        shutil.rmtree(logdir)
    writer = tf.summary.create_file_writer(logdir)

    def train_step(image_data, target):
        with tf.GradientTape() as tape:
            pred_result = model(image_data, training=True)
            giou_loss = 0
            conf_loss = 0
            prob_loss = 0

            # optimizing process
            for i in range(3):
                conv, pred = pred_result[i*2], pred_result[i*2+1]

                loss_items = yolo.compute_loss(pred, conv, *target[i], i)
                giou_loss += loss_items[0]
                conf_loss += loss_items[1]
                prob_loss += loss_items[2]

            total_loss = giou_loss + conf_loss + prob_loss

            gradients = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            tf.print("=> STEP %4d   lr: %.6f   giou_loss: %4.2f   conf_loss: %4.2f   "
                     "prob_loss: %4.2f   total_loss: %4.2f" % (global_steps, optimizer.lr.numpy(),
                                                               giou_loss, conf_loss,
                                                               prob_loss, total_loss))
            # update learning rate
            global_steps.assign_add(1)
            lr_init = 1e-3
            lr_end = 1e-6
            if global_steps < warmup_steps:
                lr = global_steps / warmup_steps * lr_init
            else:
                lr = lr_init + 0.5 * (lr_init - lr_end) * (
                    (1 + tf.cos((global_steps - warmup_steps) / (total_steps - warmup_steps) * np.pi))
                )
            optimizer.lr.assign(lr.numpy())

            # writing summary data
            with writer.as_default():
                tf.summary.scalar("lr", optimizer.lr, step=global_steps)
                tf.summary.scalar("loss/total_loss", total_loss, step=global_steps)
                tf.summary.scalar("loss/giou_loss", giou_loss, step=global_steps)
                tf.summary.scalar("loss/conf_loss", conf_loss, step=global_steps)
                tf.summary.scalar("loss/prob_loss", prob_loss, step=global_steps)
            writer.flush()

    for epoch in range(epochs):
        for image_data, target in trainset:
            train_step(image_data, target)
        model.save_weights("./yolov3")


if __name__ == "__main__":
    main()
