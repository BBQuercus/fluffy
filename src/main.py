import datetime
import luigi
import uuid

import data.dirtools
import models
import config

class ConvertImagesNpy(luigi.Task):
    
    dir_in = luigi.parameter.Parameter(default='../data/raw') 
    dir_out = luigi.parameter.Parameter(default='../data/processed') 

    def output(self):
        return {
            'train_val_images' : luigi.LocalTarget(f'{self.dir_out}/train_val_images.npy'),
            'train_val_masks' : luigi.LocalTarget(f'{self.dir_out}/train_val_masks.npy'),
            'train_images' : luigi.LocalTarget(f'{self.dir_out}/train_images.npy'),
            'train_masks' : luigi.LocalTarget(f'{self.dir_out}/train_masks.npy'),
            'val_images' : luigi.LocalTarget(f'{self.dir_out}/val_images.npy'),
            'val_masks' : luigi.LocalTarget(f'{self.dir_out}/val_masks.npy'),
            'test_images' : luigi.LocalTarget(f'{self.dir_out}/test_images.npy'),
            'test_masks' : luigi.LocalTarget(f'{self.dir_out}/test_masks.npy')
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

    config = luigi.parameter.Parameter(default=config.defaults)
    uuid = luigi.parameter.Parameter(default=str(uuid.uuid4()))
    dir_out = luigi.parameter.Parameter(default='../models') 

    def requires(self):
        return ConvertImagesNpy()

    def output(self):
        return {
            'config' : luigi.LocalTarget(f'{dir_out}/config/{uuid}.csv'),
            'graph' : luigi.LocalTarget(f'{dir_out}/graph/{uuid}.png'),
            'model' : luigi.LocalTarget(f'{dir_out}/models/{uuid}.h5'),
            'history' : luigi.LocalTarget(f'{dir_out}/history/{uuid}.csv'),
            'tensorboard' : luigi.LocalTarget(f'{dir_out}/tensorboard') # uuid subfolder?
        }
    
    def run(self):

        # Import data
        train_images = np.load(self.requires().output()['train_images'])
        train_masks = np.load(self.requires().output()['train_masks'])
        val_images = np.load(self.requires().output()['val_images'])
        val_masks = np.load(self.requires().output()['val_masks'])

        # Preprocess data
        config_preprocess = {
            'batch_size' : self.config['batch_size'],
            'scaling' : self.config['scaling'],
            'cropping' : self.config['cropping'],
            'flipping' : self.config['flipping'],
            'padding' : self.config['padding'],
            'rotation' : self.config['rotation'],
            'brightness' : self.config['brightness'],
            'contrast' : self.config['contrast'],
        }
        train_generator = data.provider.train_generator_seg(train_images, train_masks, **config_preprocess)
        val_generator = data.provider.val_generator_seg(val_images, val_masks)
        
        # Build model
        config_model = {
            'img_size' : self.config['img_size'],
            'depth' : self.config['depth'],
            'width' : self.config['width'],
            'momentum' : self.config['momentum'],
            'activation' : self.config['activation'],
            'categories' : self.config['categories'],
        }
        model = models.unet.model_seg(**config_model)

        # Compile model
        config_compile = {
            'optimizer' : self.config['optimizer'](self.config['lr']),
            'loss' : self.config['loss'],
            'metrics' : self.config['metrics'],
        }
        model.compile(**config_compile)

        # Train model
        callbacks = model.callbacks.defaults(self.output)
        config_training = {
            'x' : train_generator,
            'steps_per_epoch' : len(train_images) // self.config['batch_size'],
            'epochs' : self.config['epochs'],
            'callbacks' : callbacks,
            'validation_data' : val_generator,
            'validation_freq' : self.config['validation_freq'],
            'verbose' : self.config['verbose']
        }
        model.fit(**config_training)

        # Save final configs
        config_final = {
            **config_preprocess,
            **config_model,
            **config_compile,
            **config_training
        }
        data.dirtools.dict_to_csv(config_final, self.output()['config'])
        # config_final


# class ModelComparison(luigi.Task):

#     def requires(self):
#         return super().requires()

#     def output(self):
#         return super().output()
    
#     def run(self):
#         for delta in range(1, 11):
#             yield FetchData()
#         return super().run()

# class ModelCrossval(luigi.Task):

#     def requires(self):
#         return super().requires()

#     def output(self):
#         return super().output()
    
#     def run(self):

#         # def cross_validate(session, split_size=5):
#         #     results = []
#         #     kf = sklearn.model_selection.KFold(n_splits=split_size)
#         #     for train_idx, val_idx in kf.split(train_x_all, train_y_all):
#         #         train_x = train_x_all[train_idx]
#         #         train_y = train_y_all[train_idx]
#         #         val_x = train_x_all[val_idx]
#         #         val_y = train_y_all[val_idx]
#         #         run_train(session, train_x, train_y)
#         #         results.append(session.run(accuracy, feed_dict={x: val_x, y: val_y}))
#         #     return results

#         return super().run()