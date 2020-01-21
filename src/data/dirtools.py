import os
import glob
import random
import luigi
import numpy as np
import skimage.io


def get_file_lists_seg(path):
    '''
    Returns a two lists containing the complete file paths to the images
    or masks. To make sure things work, only sure to work file extensions
    are used.

    Args:
        path (dir): Directory containing subfolders images & masks with
            corresponding images inside.

    Returns:
        list_images (list): List containing complete file paths of images.
        list_masks (list): List containing complete file paths of masks.

    '''
    assert os.path.exists(path)

    list_images = sorted(glob.glob(f"{path}/images/*[('.tif', '.png', '.jpeg', '.stk')]"))
    list_masks = sorted(glob.glob(f"{path}/masks/*[('.tif', '.png', '.jpeg', '.stk')]"))

    assert len(list_images) == len(list_masks)

    return list_images, list_masks


def file_list_to_npy(file_list, file):
    '''
    Reads a list of files and saves them as numpy array.

    Args:
        file_list (list): List containing files to be saved.
        file (dir): File name of to be saved numpy array.

    Returns:
        None
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
    '''Shuffles two lists keeping elements in the same location respectively.
    '''

    combined = list(zip(list_x, list_y))
    random.shuffle(combined)
    return zip(*combined)


def train_valid_split(list_x, list_y, valid_split=0.25, shuffle=True):
    '''
    Shuffles and splits two lists into train / validation sets.

    Args:
        list_x (list): First list.
        list_y (list): Second list.
        valid_split (float): Percentage split to be used for validation.
        shuffle (bool): Decide if lists should be shuffled beforehand.
    
    Returns:
        train_x (list): Train list containing 1-valid_split % elements.
        train_y (list): Train list containing 1-valid_split % elements.
        valid_x (list): Valid list containing valid_slit % elements.
        valid_y (list): Valid list containing valid_slit % elements.
    '''

    if shuffle:
        list_x, list_y = _shuffle_lists(list_x, list_y)

    split_len = round(len(list_x) * valid_split)
    train_x, valid_x = list_x[split_len:], list_x[:split_len]
    train_y, valid_y = list_y[split_len:], list_y[:split_len]

    return train_x, train_y, valid_x, valid_y


def dict_to_csv(input_dict, file):
    '''
    Converts a dictionary to a csv file.

    Args:
        input_dict (dict): Dictionary to be saved.
        file (dir): Filename of to be saved csv file.

    Returns:
        None
    '''
    assert type(input_dict) == dict
    assert file.endswith('.csv')

    with open(file, 'w') as f:
        f.write(f'Parameters used\n')
        for key in input_dict.keys():
            f.write(f'{key}, {input_dict[key]}\n')

    return None