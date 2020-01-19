import itertools
import datetime
import luigi
import numpy as np
import tensorflow as tf

import data.dirtools
import data.provider
import models.unet
import models.resunet
import models.metrics
import models.callbacks


class ConvertImagesNpy(luigi.Task):
    '''
    Converts images and masks for training/validation and testing
    into numpy files for easier access later on. Does not do any
    preprocessing or file manipulation.

    Args:
        dir_in (dir): Path to data files in the following format.
            Brackets denote multiple possibilties.
            dir_in/(train_val, test)/(images, masks)/(...jpg/tif/...)
        dir_out (dir): Path to where numpy files should be saved.
    '''

    dir_in = luigi.parameter.Parameter(default='../data/raw')
    dir_out = luigi.parameter.Parameter(default='../data/processed')

    def output(self):
        return {
            'train_val_images': luigi.LocalTarget(f'{self.dir_out}/train_val_images.npy'),
            'train_val_masks': luigi.LocalTarget(f'{self.dir_out}/train_val_masks.npy'),
            'train_images': luigi.LocalTarget(f'{self.dir_out}/train_images.npy'),
            'train_masks': luigi.LocalTarget(f'{self.dir_out}/train_masks.npy'),
            'val_images': luigi.LocalTarget(f'{self.dir_out}/val_images.npy'),
            'val_masks': luigi.LocalTarget(f'{self.dir_out}/val_masks.npy'),
            'test_images': luigi.LocalTarget(f'{self.dir_out}/test_images.npy'),
            'test_masks': luigi.LocalTarget(f'{self.dir_out}/test_masks.npy')
        }

    def run(self):

        # Import lists
        train_val_images, train_val_masks = data.dirtools.get_file_lists_seg(f'{self.dir_in}/train_val')
        test_images, test_masks = data.dirtools.get_file_lists_seg(f'{self.dir_in}/test')

        # Train / validation split
        train_images, train_masks, val_images, val_masks = data.dirtools.train_valid_split(train_val_images, train_val_masks)

        # Save to numpy
        data.dirtools.file_list_to_npy(train_val_images, self.output()['train_val_images'])
        data.dirtools.file_list_to_npy(train_val_masks, self.output()['train_val_masks'])
        data.dirtools.file_list_to_npy(train_images, self.output()['train_images'])
        data.dirtools.file_list_to_npy(train_masks, self.output()['train_masks'])
        data.dirtools.file_list_to_npy(val_images, self.output()['val_images'])
        data.dirtools.file_list_to_npy(val_masks, self.output()['val_masks'])
        data.dirtools.file_list_to_npy(test_images, self.output()['test_images'])
        data.dirtools.file_list_to_npy(test_masks, self.output()['test_masks'])


class TrainOneModel(luigi.Task):
    '''
    Trains one model from start to end including data preprocessing,
    model generation, validation, training. Does not include
    hyperparameter tuning, crossvalidation, score reporting.

    Args:
        config (dict): Dictionary containing all configurations.
        uuid (str): "Unique User ID" or unique identifier for naming.
        dir_out (dir): Path where model and logs should be saved.
    '''

    # Basic
    uuid = luigi.parameter.Parameter(default=datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    dir_out = luigi.parameter.Parameter(default='../models')
    # hparams = luigi.parameter.Parameter(default=None)

    # Architecture
    resnet = luigi.parameter.BoolParameter(default=False)
    img_size = luigi.parameter.IntParameter(default=256)
    depth = luigi.parameter.IntParameter(default=4)
    width = luigi.parameter.IntParameter(default=16)
    classes = luigi.parameter.BoolParameter(default=True)

    # Compilation – Luigi does not allow for objects as parameters
    lr = luigi.parameter.FloatParameter(default=0.001)
    loss = luigi.parameter.Parameter()

    # Training
    epochs = luigi.parameter.IntParameter(default=150)
    validation_freq = luigi.parameter.IntParameter(default=1)
    batch_size = luigi.parameter.IntParameter(default=16)

    # Data
    bit_depth = luigi.parameter.IntParameter(default=16)
    border = luigi.parameter.BoolParameter(default=True)
    border_size = luigi.parameter.IntParameter(default=2)
    touching_only = luigi.parameter.BoolParameter(default=True)

    def requires(self):
        return ConvertImagesNpy()

    def output(self):
        return {
            'config': luigi.LocalTarget(f'{self.dir_out}/config/{self.uuid}.csv'),
            'graph': luigi.LocalTarget(f'{self.dir_out}/graph/{self.uuid}.png'),
            'model_best': luigi.LocalTarget(f'{self.dir_out}/models/{self.uuid}.h5'),
            'model_final': luigi.LocalTarget(f'{self.dir_out}/models/{self.uuid}_final.h5'),
            'history': luigi.LocalTarget(f'{self.dir_out}/history/{self.uuid}.csv'),
            'tensorboard': luigi.LocalTarget(f'{self.dir_out}/tensorboard/{self.uuid}/')
        }

    def run(self):

        # Import data
        train_images = np.load(self.input()['train_images'].path, allow_pickle=True)
        train_masks = np.load(self.input()['train_masks'].path, allow_pickle=True)
        val_images = np.load(self.input()['val_images'].path, allow_pickle=True)
        val_masks = np.load(self.input()['val_masks'].path, allow_pickle=True)

        # Prepare generators
        config_preprocess = {
            'img_size': self.img_size,
            'bit_depth': self.bit_depth,
            'batch_size': self.batch_size,
            'border': self.border,
            'border_size': self.border_size,
            'touching_only': self.touching_only
        }
        train_generator = data.provider.generator_seg(train_images, train_masks, training=True, **config_preprocess)
        val_generator = data.provider.generator_seg(val_images, val_masks, training=False, **config_preprocess)

        # Build model
        config_model = {
            'depth': self.depth,
            'width': self.width,
            'classes': self.classes,
        }
        model = models.unet.model_seg(**config_model)
        tf.keras.utils.plot_model(model, to_file=self.output()['graph'].path, show_shapes=True)
        model.summary()

        # Compile model
        config_compile = {
            'optimizer': tf.keras.optimizers.Adam(self.lr),
            'loss': self.loss,
            'metrics': models.metrics.default(),
        }
        model.compile(**config_compile)

        # Train model
        callbacks = models.callbacks.defaults(
            self.output()['model_best'].path,
            self.output()['history'].path,
            self.output()['tensorboard'].path
        )
        config_training = {
            'steps_per_epoch': len(train_images) // self.batch_size,
            'epochs': self.epochs,
            'callbacks': callbacks,
            'validation_data': val_generator,
            'validation_freq': self.validation_freq,
            'validation_steps': len(val_images) // self.batch_size,
        }
        model.fit_generator(train_generator, **config_training)
        model.save(self.output()['model_final'].path)

        # Save final configs
        config_final = {
            **config_preprocess,
            **config_model,
            **config_compile,
            **config_training
        }
        data.dirtools.dict_to_csv(config_final, self.output()['config'].path)


def main():
    hparam_lr = [0.0001]
    hparam_bs = [16, 32, 64]
    hparam_width = [16, 64]
    hparam_depth = [4]
    hparam_loss = [tf.keras.losses.CategoricalCrossentropy(), tf.keras.losses.CosineSimilarity(), tf.keras.losses.MeanSquaredError()]

    for lr, width, depth, batch_size, loss in itertools.product(
            hparam_lr, hparam_width, hparam_depth, hparam_bs, hparam_loss):

        config_new = {
            'lr': lr,
            'width': width,
            'depth': depth,
            'batch_size': batch_size,
        }

        uuid = f'lr-{lr}_w-{width}_d-{depth}_bs-{batch_size}_loss-{hparam_loss.index(loss)}'

        luigi.build([TrainOneModel(uuid, loss=loss, **config_new)], local_scheduler=True)


if __name__ == "__main__":
    main()


# def mainhp():
#     import tensorboard.plugins.hparams.api as hp
#     HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([16, 32]))
#     HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.1, 0.2))
#     HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd']))
#     METRIC_ACCURACY = 'accuracy'

#     with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
#         hp.hparams_config(
#             hparams=[HP_NUM_UNITS, HP_DROPOUT, HP_OPTIMIZER],
#             metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],
#         )

#     def train_test_model(hparams):
#         model = tf.keras.models.Sequential([
#             tf.keras.layers.Flatten(),
#             tf.keras.layers.Dense(hparams[HP_NUM_UNITS], activation=tf.nn.relu),
#             tf.keras.layers.Dropout(hparams[HP_DROPOUT]),
#             tf.keras.layers.Dense(10, activation=tf.nn.softmax),
#         ])
#         model.compile(
#             optimizer=hparams[HP_OPTIMIZER],
#             loss='sparse_categorical_crossentropy',
#             metrics=['accuracy'],
#         )

#         model.fit(x_train, y_train, epochs=1) # Run with 1 epoch to speed things up for demo purposes
#         _, accuracy = model.evaluate(x_test, y_test)
#         return accuracy

#     def run(run_dir, hparams):
#         with tf.summary.create_file_writer(run_dir).as_default():
#             hp.hparams(hparams)  # record the values used in this trial
#             accuracy = train_test_model(hparams)
#             tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)

#     model.fit(
#         ...,
#         callbacks=[
#             tf.keras.callbacks.TensorBoard(logdir),  # log metrics
#             hp.KerasCallback(logdir, hparams),  # log hparams
#         ],
#     )

#     for num_units in HP_NUM_UNITS.domain.values:
#         for dropout_rate in (HP_DROPOUT.domain.min_value, HP_DROPOUT.domain.max_value):
#             for optimizer in HP_OPTIMIZER.domain.values:
#                 hparams = {
#                     HP_NUM_UNITS: num_units,
#                     HP_DROPOUT: dropout_rate,
#                     HP_OPTIMIZER: optimizer,
#                 }
#                 run_name = "run-%d" % session_num
#                 print('--- Starting trial: %s' % run_name)
#                 print({h.name: hparams[h] for h in hparams})
#                 run('logs/hparam_tuning/' + run_name, hparams)
#                 session_num += 1
