import glob
import json
import logging
import numpy as np
import os
import pickle
import random
import skopt

LOG_FORMAT = '%(levelname)s %(asctime)s - %(filename)s %(funcName)s %(lineno)s - %(message)s'
logging.basicConfig(filename='./train_model.log',
                    level=logging.DEBUG,
                    format=LOG_FORMAT,
                    filemode='a')
log = logging.getLogger()


def shuffle_lists(x_list, y_list):
    ''' Shuffles two lists keeping items in the same relative locations. '''
    if not all(isinstance(i, list) for i in [x_list, y_list]):
        raise TypeError(f'x_list, y_list must be list but is {type(x_list)}, {type(y_list)}.')

    combined = list(zip(x_list, y_list))
    random.shuffle(combined)
    tuple_x, tuple_y = zip(*combined)
    return list(tuple_x), list(tuple_y)


def train_valid_split(x_list, y_list, valid_split=0.25):
    '''
    Splits two lists (images and masks) into random training and validation sets.

    Args:
        - x_list (list): List containing filenames of all images.
        - y_list (list): List containing filenames of all masks.
        - valid_split (float, optional): Number between 0-1 to denote
            the percentage of examples used for validation.
    Returns:
        - x_train, x_valid, y_train, y_valid (lists): Splited lists
            containing training or validation examples respectively.
    '''
    if not all(isinstance(i, list) for i in [x_list, y_list]):
        raise TypeError(f'x_list, y_list must be list but is {type(x_list)}, {type(y_list)}.')
    if not any(isinstance(valid_split, i) for i in [int, float]):
        raise TypeError(f'valid_split must be float but is a {type(valid_split)}.')
    if not all(isinstance(i, str) for i in x_list):
        raise TypeError(f'x_list items must be str but are {type(x_list[0])}.')
    if not all(isinstance(i, str) for i in x_list):
        raise TypeError(f'x_list items must be str but are {type(x_list[0])}.')
    if len(x_list) != len(y_list):
        raise ValueError(f'Lists must be of equal length: {len(x_list)} != {len(y_list)}.')
    if any(len(i) == 0 for i in x_list):
        raise ValueError('x_list items must not be empty.')
    if any(len(i) == 0 for i in y_list):
        raise ValueError('y_list items must not be empty.')
    if len(x_list) <= 2:
        raise ValueError('Lists must contain 2 elements or more.')
    if not 0 <= valid_split <= 1:
        raise ValueError(f'valid_split must be between 0-1 but is {valid_split}.')

    split_len = round(len(x_list) * valid_split)

    x_list, y_list = shuffle_lists(x_list, y_list)
    x_valid, x_train = x_list[:split_len], x_list[split_len:]
    y_valid, y_train = y_list[:split_len], y_list[split_len:]

    return x_train, x_valid, y_train, y_valid


def get_k_folds_indices(k, x_list):
    '''
    Returns the list of indices for k train/val splits.
    Args:
        - k (int): Number of splits.
        - x_list (list): Full length train/val list to be split.
    Returns:
        - list_k_folds (list):
    '''
    if not isinstance(k, int):
        raise TypeError(f'k must be int but is {type(k)}.')
    if not isinstance(x_list, list):
        raise TypeError(f'x_list must be int but is {type(x_list)}.')
    if k == 0:
        raise ZeroDivisionError('k must not be zero.')

    def __inverse_selection(list_full, list_selection):
        ''' Returns the not selected items of a list. '''
        return [i for i in list_full if i not in list_selection]

    # Numpy w/ list conversion is faster than list only
    bin_size = len(x_list) // k
    list_full = np.arange(len(x_list))
    list_train = []
    list_valid = []

    index_start = 0
    for i in range(k):
        list_selection = list_full[index_start:index_start+bin_size]
        list_inverse = __inverse_selection(list_full, list_selection)
        list_valid.append(list(list_selection))
        list_train.append(list(list_inverse))
        index_start += bin_size

    list_k_folds = list(zip(list_train, list_valid))
    return list_k_folds


def split_k_folds(k, x_list, y_list):
    '''
    Returns split lists containing x_train, y_train, x_valid, y_valid per k fold.
    '''
    if not all(isinstance(i, list) for i in [x_list, y_list]):
        raise TypeError(f'x_list, y_list must be int but is {[type(i) for i in [x_list, y_list]]}.')

    k_fold_indices = get_k_folds_indices(k, x_list)
    k_folds = []

    for train_ind, valid_ind in k_fold_indices:
        x_train = list(np.take(x_list, train_ind))
        y_train = list(np.take(y_list, train_ind))
        x_valid = list(np.take(x_list, valid_ind))
        y_valid = list(np.take(y_list, valid_ind))
        k_folds.append([x_train, y_train, x_valid, y_valid])

    return k_folds


def get_file_lists(path):
    ''' Imports file lists within the standard images/masks/ format. '''
    if not isinstance(path, str):
        raise TypeError(f'path must be str but is {type(path)}.')
    if not os.path.exists(path):
        raise ValueError(f'path must exist: {path} does not.')

    list_images = sorted(glob.glob(f'{path}/images/*.tif'))
    list_masks = sorted(glob.glob(f'{path}/masks/*.tif'))

    if not len(list_images) == len(list_masks):
        raise ValueError(f'list_images and list_masks must be same length: {len(list_images)} != {len(list_masks)}.')
    if not list_images:
        raise ValueError(f'list_images, list_masks must not be empty.')

    return list_images, list_masks


def list_to_pickle(fname, x_list):
    ''' Saves lists to a pickle object. '''
    if not isinstance(fname, str):
        raise TypeError(f'fname must be str but is {type(fname)}.')
    if not isinstance(x_list, list):
        raise TypeError(f'x_list must be dict but is {type(x_list)}.')
    if os.path.exists(fname):
        log.warning(f'fname "{fname}" already exists. Will be overwritten.')

    with open(fname, 'wb') as f:
        pickle.dump(x_list, f)


def pickle_to_list(fname):
    ''' Opens pickled lists. '''
    if not isinstance(fname, str):
        raise TypeError(f'fname must be str but is {type(fname)}.')
    if not os.path.exists(fname):
        raise ValueError(f'fname must exist: {fname} does not.')

    with open(fname, 'rb') as f:
        x_list = pickle.load(f)
        assert isinstance(x_list, list)
    return x_list


def skopt_to_pickle(fname, result):
    ''' Saves skopt results as pickled objects. '''
    if not isinstance(fname, str):
        raise TypeError(f'fname must be str but is {type(fname)}.')
    if os.path.exists(fname):
        log.warning(f'fname "{fname}" already exists. Will be overwritten.')

    with open(fname, 'wb') as f:
        skopt.dump(result, f, False)


def dict_to_json(fname, x_dict):
    ''' Saves a python dictionary as json file format. '''
    if not isinstance(fname, str):
        raise TypeError(f'fname must be str but is {type(fname)}.')
    if not isinstance(x_dict, dict):
        raise TypeError(f'x_dict must be dict but is {type(x_dict)}.')
    if os.path.exists(fname):
        log.warning(f'fname "{fname}" already exists. Will be overwritten.')

    with open(fname, 'w') as f:
        json.dump(x_dict, f)


def json_to_dict(fname):
    ''' Opens json files as python dictionaries. '''
    if not isinstance(fname, str):
        raise TypeError(f'fname must be str but is {type(fname)}.')
    if not os.path.exists(fname):
        raise ValueError(f'fname must exist: {fname} does not.')

    with open(fname, 'r') as f:
        x_dict = json.load(f)
        assert isinstance(x_dict, dict)
    return x_dict
