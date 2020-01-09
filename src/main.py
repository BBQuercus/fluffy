'''
Images to numpy
Parameters to json / pkl
Raw model to pb / h5
Train one model w/ raw and params
Data preprocessing to generator – save as?
HParam search
Crossvalidation
Model comparison
Visual output
Validate on testing data
'''

import datetime
import luigi

import data
import models

class ConvertImagesPkl(luigi.Task):
    
    dir_in = luigi.parameter(default='../data/raw') 
    dir_out = luigi.parameter(default='../data/processed') 
    run_id = luigi.parameter(default=str(datetime.date.today()))

    def output(self):
        return {'train_val_images':luigi.LocalTarget(f'{self.dir_out}/{self.run_id}_train_val_images.npy'),
                'train_val_masks':luigi.LocalTarget(f'{self.dir_out}/{self.run_id}_train_val_masks.npy'),
                'train_images':luigi.LocalTarget(f'{self.dir_out}/{self.run_id}_train_images.npy'),
                'train_masks':luigi.LocalTarget(f'{self.dir_out}/{self.run_id}_train_masks.npy'),
                'val_images':luigi.LocalTarget(f'{self.dir_out}/{self.run_id}_val_images.npy'),
                'val_masks':luigi.LocalTarget(f'{self.dir_out}/{self.run_id}_val_masks.npy'),
                'test_images':luigi.LocalTarget(f'{self.dir_out}/{self.run_id}_test_images.npy'),
                'test_masks':luigi.LocalTarget(f'{self.dir_out}/{self.run_id}_test_masks.npy')}

    def run(self):

        # Import lists
        train_val_images, train_val_masks = data.dirtools.get_file_lists_segmentation(f'{root}/train_val')
        test_images, test_masks = data.dirtools.get_file_lists_segmentation(f'{root}/test')

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

    config = luigi.parameter.Parameter(default=models.config.defaults)

    def requires(self):
        for delta in range(1, 11):
            yield FetchData()
        yield ConvertImagesPkl()

    def output(self):
        # config
        # trained model
        # logs
        # simple comparison metrics
        return None
    
    def run(self):
        # import hparams
        # build model
        # import data / preprocess
        # train model
        return None

class ModelComparison(luigi.Task):

    def requires(self):
        return super().requires()

    def output(self):
        return super().output()
    
    def run(self):
        return super().run()

class ModelCrossval(luigi.Task):

    def requires(self):
        return super().requires()

    def output(self):
        return super().output()
    
    def run(self):

        # def cross_validate(session, split_size=5):
        #     results = []
        #     kf = sklearn.model_selection.KFold(n_splits=split_size)
        #     for train_idx, val_idx in kf.split(train_x_all, train_y_all):
        #         train_x = train_x_all[train_idx]
        #         train_y = train_y_all[train_idx]
        #         val_x = train_x_all[val_idx]
        #         val_y = train_y_all[val_idx]
        #         run_train(session, train_x, train_y)
        #         results.append(session.run(accuracy, feed_dict={x: val_x, y: val_y}))
        #     return results

        return super().run()

def main():
    # Cmd line args if hparam search
    return None

if __name__ == "__main__":
    main()