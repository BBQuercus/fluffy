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

    # Data
    border = luigi.parameter.BoolParameter(default=True)
    border_size = luigi.parameter.IntParameter(default=2)
    add_touching = luigi.parameter.BoolParameter(default=False)

    # Architecture
    resnet = luigi.parameter.BoolParameter(default=False)
    img_size = luigi.parameter.IntParameter(default=256)
    depth = luigi.parameter.IntParameter(default=4)
    width = luigi.parameter.IntParameter(default=8)
    momentum = luigi.parameter.FloatParameter(default=0.9)
    dropout = luigi.parameter.FloatParameter(default=0.2)
    # TODO change automatically w/ borders
    classes = luigi.parameter.IntParameter(default=3)

    # Compilation - Luigi does not allow for objects as parameters
    lr = luigi.parameter.FloatParameter(default=0.0001)
    loss = luigi.parameter.Parameter()

    # Training
    epochs = luigi.parameter.IntParameter(default=150)
    validation_freq = luigi.parameter.IntParameter(default=1)
    batch_size = luigi.parameter.IntParameter(default=16)

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
            'batch_size': self.batch_size,
            'border': self.border,
            'border_size': self.border_size,
            'add_touching': self.add_touching
        }
        train_generator = data.provider.generator_seg(train_images, train_masks, training=True, **config_preprocess)
        val_generator = data.provider.generator_seg(val_images, val_masks, training=False, **config_preprocess)

        # Build model
        config_model = {
            'depth': self.depth,
            'width': self.width,
            'momentum': self.momentum,
            'dropout': self.dropout,
            'classes': self.classes
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
    '''
    '''
    # hparam_lr = [1e-4, 1e-5]
    # for lr in hparam_lr:
    lr = 1e-4
    loss = tf.keras.losses.CategoricalCrossentropy()
    uuid = f'aug-0_lr-{lr}'
    luigi.build([TrainOneModel(uuid, loss=loss, lr=lr)], local_scheduler=True)


if __name__ == "__main__":
    main()


# class CrossValidateModel(luigi.Task):

#     hparam_... = luigi.parameter.Parameter()
#     kfold = luigi.parameter.Parameter(default=5)

#     def requires(self):
#         return ConvertImagesNpy()

#     def output(self):
#         return {
#             'model': [],
#         }

#     def run(self):
#         train_val_images = np.load(self.input()['train_val_images'].path, allow_pickle=True)
#         train_val_masks = np.load(self.input()['train_val_images'].path, allow_pickle=True)

#         kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
#         cvscores = []
#         for train, val in kfold.split(X, Y):
#             scores = model.evaluate(X[val], Y[val], verbose=0)
#             print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#             cvscores.append(scores[1] * 100)
#         return None