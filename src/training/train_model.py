import logging
import metaflow
import numpy as np
import os
import skopt
import datetime
import tensorflow as tf

import models
import dirtools
import data
import metrics


LOG_FORMAT = '%(levelname)s %(asctime)s - %(filename)s %(funcName)s %(lineno)s - %(message)s'
logging.basicConfig(filename='./train_model.log',
                    level=logging.DEBUG,
                    format=LOG_FORMAT,
                    filemode='a')
log = logging.getLogger()


def train_model(
    x_train,
    x_valid,
    y_train,
    y_valid,
    binary,
    optimizer,
    learning_rate,
    batch_size,
    model_depth,
    model_width,
    last_width,
    epochs,
    callbacks=[],
        ):
    if not all(isinstance(i, list) for i in [x_train, x_valid, y_train, y_valid, callbacks]):
        raise TypeError(f'''x_train, x_valid, y_train, y_valid, callbacks must be list but are
                            {[type(i) for i in [x_train, x_valid, y_train, y_valid, callbacks]]}.''')
    if not isinstance(binary, bool):
        raise TypeError(f'binary must be bool but is {type(binary)}.')
    if not isinstance(learning_rate, float):
        raise TypeError(f'learning_rate must be float but is {type(learning_rate)}.')
    if not all(isinstance(i, int) for i in [batch_size, model_depth, model_width, last_width, epochs]):
        raise TypeError(f'''batch_size, model_depth, model_width, last_width, epochs must be int but are
                            {[type(i) for i in [batch_size, model_depth, model_width, last_width, epochs]]}''')

    # Model Building
    model = models.unet(binary=binary,
                        depth=model_depth,
                        model_width=model_width,
                        last_width=last_width)

    # Model Compiling
    if binary:
        model_loss = tf.keras.losses.binary_crossentropy
        model_metrics = [tf.keras.metrics.binary_accuracy, metrics.dice_coef]
    else:
        model_loss = tf.keras.losses.categorical_crossentropy
        model_metrics = [tf.keras.metrics.categorical_accuracy, metrics.dice_coef]

    optimizer = optimizer(lr=learning_rate)

    model.compile(loss=model_loss, metrics=model_metrics, optimizer=optimizer)

    # Data Generators
    train_gen = data.random_sample_generator(
        x_list=x_train,
        y_list=y_train,
        binary=binary,
        augment=True,
        batch_size=int(batch_size),
        bit_depth=16,
        crop_size=256)
    val_gen = data.random_sample_generator(
        x_list=x_valid,
        y_list=y_valid,
        binary=binary,
        augment=False,
        batch_size=16,
        bit_depth=16,
        crop_size=256)

    # Training
    history = model.fit_generator(
                generator=train_gen,
                validation_data=val_gen,
                validation_steps=20,
                callbacks=callbacks,
                epochs=epochs,
                steps_per_epoch=20,
                verbose=2)

    del model
    tf.keras.backend.clear_session()

    return history


class Fluffy(metaflow.FlowSpec):
    '''
    Fluffy training with optional hyperparameter search.
    '''

    dir_in = metaflow.Parameter(
        "dir_in", help="Path to data files."
    )
    dir_out = metaflow.Parameter(
        "dir_out", help="Path to where training files should be saved."
    )
    name = metaflow.Parameter(
        "name", help="Name of model (prefix for all files).", default="model", type=str
    )
    binary = metaflow.Parameter(
        "binary", help="If model is binary or categorical.", default=True, type=bool
    )
    use_defaults = metaflow.Parameter(
        "use_defaults",
        help="If hparam optimization should be performed or default params should be used.",
        default=False,
        type=bool,
    )
    n_splits = metaflow.Parameter(
        "n_splits", help="Number of splits/k for crossvalidation.", default=5, type=int
    )

    default_hparams = {
        'optimizer': tf.keras.optimizers.Adam,
        'learning_rate': 0.0001,
        'batch_size': 16,
        'model_depth': 4,
        'model_width': 16,
        'last_width': 16,
    }

    @metaflow.step
    def start(self):
        '''
        Initial placeholder start step.
        Sets up the output directory.
        '''
        os.makedirs(self.dir_out, exist_ok=True)

        self.basename = f'{self.dir_out}/{datetime.date.today().strftime("%Y%m%d")}_{self.name}'
        self.next(self.train_test_split)

    @metaflow.step
    def train_test_split(self):
        '''
        Splits the entire dataset into train/valid and test set.
        Train/valid will be used for hyperparameter optimization.
        Test will be used to avoid overfitting in the final training step.
        If "use_defaults" was selected, will skip over optimization.
        Does not do any preprocessing or file manipulation.
        '''
        # Import lists
        images, masks = dirtools.get_file_lists(self.dir_in)
        log.info(f'Full dataset: images-{len(images)}, masks-{len(masks)}')

        # Train / test split
        train_val_images, test_images, \
            train_val_masks, test_masks = dirtools.train_valid_split(images, masks, valid_split=0.2)
        log.info(f'Train/test split: train_val-{len(train_val_images)}, test-{len(test_images)}')

        # Save to pickle
        dirtools.list_to_pickle(f'{self.basename}_train_val.pkl', [train_val_images, train_val_masks])
        dirtools.list_to_pickle(f'{self.basename}_test.pkl', [test_images, test_masks])
        log.info(f'Train/test split saved as pickle in {self.dir_out}.')

        # Package
        self.train_val_images = train_val_images
        self.train_val_masks = train_val_masks
        self.test_images = test_images
        self.test_masks = test_masks
        self.next(self.crossval_split)

    @metaflow.step
    def crossval_split(self):
        '''
        Splits the train/valid dataset into k_folds to be used in hparam optimization.
        '''
        k_folds = dirtools.split_k_folds(self.n_splits, self.train_val_images, self.train_val_masks)
        log.info(f'K_folds prepared: {len(k_folds)} with {len(k_folds[0][0])}/{len(k_folds[0][3])}.')

        self.k_folds = k_folds
        self.next(self.hparam_optimizer)

    @metaflow.step
    def hparam_optimizer(self):
        '''
        Performs bayesian hyperparameter optimization on the train_val data using crossvalidation.
        Saves the best hyperparameters for usage during the final training step.
        Does not save the model.
        '''

        # Define HParam search space
        hp_optimizer = skopt.space.Categorical([tf.keras.optimizers.RMSprop, tf.keras.optimizers.Adam], name='optimizer')
        hp_learning_rate = skopt.space.Categorical([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1], name='learning_rate')
        hp_batch_size = skopt.space.Categorical([1, 2, 4, 8, 16, 32], name='batch_size')
        hp_model_depth = skopt.space.Categorical([1, 2, 3, 4, 5], name='model_depth')
        hp_model_width = skopt.space.Categorical([1, 2, 4, 8, 16], name='model_width')
        hp_last_width = skopt.space.Categorical([8, 16, 32], name='last_width')
        dimensions = [hp_optimizer, hp_learning_rate, hp_batch_size, hp_model_depth, hp_model_width, hp_last_width]
        default_parameters = list(self.default_hparams.values())
        log.info(f'''
            HParam search space:
                optimizer-{hp_optimizer.categories},
                learning_rate-{hp_learning_rate.categories},
                batch_size-{hp_batch_size.categories},
                model_depth-{hp_model_depth.categories},
                model_width-{hp_model_width.categories},
                last_width-{hp_last_width.categories}
            ''')

        # Optimizer function
        @skopt.utils.use_named_args(dimensions=dimensions)
        def optimizer(optimizer, learning_rate, batch_size, model_depth, model_width, last_width):
            log.info(f'Starting optimizer with {locals()}.')

            accuracy = []
            for n, (x_train, y_train, x_valid, y_valid) in enumerate(self.k_folds):
                log.info(f'Starting k_fold #{n}.')
                history = train_model(
                    x_train=x_train,
                    x_valid=x_valid,
                    y_train=y_train,
                    y_valid=y_valid,
                    binary=self.binary,

                    optimizer=optimizer,
                    learning_rate=learning_rate,
                    batch_size=batch_size,
                    model_depth=model_depth,
                    model_width=model_width,
                    last_width=last_width,

                    epochs=10
                )
                k_fold_accuracy = models.get_accuracy(history)
                log.info(f'Finished k_fold #{n} with an accuracy of {k_fold_accuracy}.')
                accuracy.append(k_fold_accuracy)

            accuracy = np.mean(accuracy)
            log.info(f'Finished optimizer with an accuracy of {accuracy}.')
            return 1-accuracy

        if not self.use_defaults:
            opt_result = skopt.gp_minimize(optimizer, dimensions,
                                           x0=default_parameters,
                                           acq_optimizer='auto',
                                           n_calls=12)
            log.info(f'Optimization finished with the best score of {1-abs(opt_result.fun)}.')
            dirtools.skopt_to_pickle(f'{self.basename}.gz', opt_result)
            self.opt_hparams = {
                'optimizer': opt_result.x[0],
                'learning_rate': opt_result.x[1],
                'batch_size': opt_result.x[2],
                'model_depth': opt_result.x[3],
                'model_width': opt_result.x[4],
                'last_width': opt_result.x[5]
            }

        self.next(self.train_final_model)

    @metaflow.step
    def train_final_model(self):
        '''
        Main step to train the final model.
        Uses the test dataset as validation to avoid overfitting.
        '''
        if self.use_defaults:
            hparams = self.default_hparams
        else:
            hparams = self.opt_hparams

        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(f'{self.basename}.h5', save_best_only=True),
            tf.keras.callbacks.CSVLogger(filename=f'{self.basename}.csv'),
            tf.keras.callbacks.TensorBoard(self.basename)
        ]
        _ = train_model(
            x_train=self.train_val_images,
            x_valid=self.test_images,
            y_train=self.train_val_masks,
            y_valid=self.test_masks,
            binary=self.binary,

            optimizer=hparams['optimizer'],
            learning_rate=hparams['learning_rate'],
            batch_size=hparams['batch_size'],
            model_depth=hparams['model_depth'],
            model_width=hparams['model_width'],
            last_width=hparams['last_width'],

            epochs=250,
            callbacks=callbacks
        )

        self.next(self.end)

    @metaflow.step
    def end(self):
        '''
        Final placeholder end step.
        '''
        pass


if __name__ == "__main__":
    Fluffy()
