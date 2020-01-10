import os
import glob
import random
import csv
import luigi
import numpy as np
import skimage.io


def get_file_lists_seg(root):
    '''
    '''
    assert os.path.exists(root)

    list_images = glob.glob(f"{root}/images/*[('.tif', '.png', '.jpeg', '.stk')]")
    list_masks = glob.glob(f"{root}/masks/*[('.tif', '.png', '.jpeg', '.stk')]")

    return list_images, list_masks

def file_list_to_npy(file_list, file):
    '''
    '''
    if type(file) == luigi.local_target.LocalTarget:
        file = file.path
    assert file.endswith('.npy')
    for f in file_list:
        assert f.endswith((('.tif', '.png', '.jpeg', '.stk')))

    list_images = list(map(skimage.io.imread, file_list))
    list_images = np.array(list_images)
    list_images.dump(file)

    return None

def _shuffle_lists(list_x, list_y):

    combined = list(zip(list_x, list_y))
    random.shuffle(combined)
    return zip(*combined)

def train_valid_split(list_x, list_y, valid_split=0.25, shuffle=True):
    '''
    '''

    if shuffle:
        list_x, list_y = _shuffle_lists(list_x, list_y)

    split_len = round(len(list_x) * valid_split)
    train_x, valid_x = list_x[split_len:], list_x[:split_len]
    train_y, valid_y = list_y[split_len:], list_y[:split_len]

    return train_x, train_y, valid_x, valid_y

def dict_to_csv(input_dict, file):
    with open(file, 'w') as f:
        for key in input_dict.keys():
            f.write(f'{key}, {input_dict[key]}')