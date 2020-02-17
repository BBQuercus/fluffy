'''
Populates a directory substructure at a specified location.
Used to convert various raw images into the format used to label.
The second step in the labeling workflow.
'''

import click
import czifile
import glob
import logging
import ntpath
import numpy as np
import os
import skimage.io

EXTENSIONS = ['.czi', '.stk', '.tiff', '.tif', '.jpeg', '.jpg', '.png']
LOG_FORMAT = '%(levelname)s %(asctime)s - %(filename)s %(funcName)s %(lineno)s - %(message)s'
logging.basicConfig(filename='./data.log',
                    level=logging.DEBUG,
                    format=LOG_FORMAT,
                    filemode='a')
log = logging.getLogger()


def adaptive_imread(fname):
    ''' Opens images depending on filetype. '''
    if fname.endswith(EXTENSIONS[0]):
        return czifile.imread(fname)
    if any(fname.endswith(i) for i in EXTENSIONS[1:]):
        return skimage.io.imread(fname)
    raise ValueError(f'File {fname} not of of type {EXTENSIONS}')


def get_min_axis(image):
    ''' Returns the index of a smallest axis of an image. '''
    shape = image.shape
    axis = shape.index(min(shape))
    return axis


def process_timelapse(image):
    '''
    Returns the middle time frame from a timelapse movie.
    Assumes that the smalles axis is the time axis.
    '''
    image = image.squeeze()
    axis = get_min_axis(image)
    index = image.shape[axis] // 2
    return np.take(image, index, axis=axis)


def process_zstack(image):
    '''
    Returns the maximum projection from a zstack.
    Assumes that the smalles axis is the z axis.
    '''
    image = image.squeeze()
    axis = get_min_axis(image)
    return np.max(image, axis=axis)


def process_standard(image):
    ''' Returns squeezed image. '''
    return image.squeeze()


def save_image(dir_base, basename, image):
    ''' Saves image in the proper format. '''
    dir_base = os.path.join(dir_base, 'Labeling', basename)
    os.makedirs(dir_base, exist_ok=True)
    dir_images = os.path.join(dir_base, 'images')
    os.makedirs(dir_images, exist_ok=True)
    dir_masks = os.path.join(dir_base, 'masks')
    os.makedirs(dir_masks, exist_ok=True)
    log.info(f'Subdirs for image {basename} created.')
    skimage.io.imsave(f'{dir_images}/{basename}.tif', image)
    log.info(f'Image saved.')


@click.command()
@click.option('--dir_base',
              type=click.Path(exists=True),
              prompt='Path to the base directory',
              required=True,
              help='Path to the base directory.')
@click.option('--allslices',
              prompt='Use all slices',
              is_flag=True,
              help='If all slices / time points should be converted.')
@click.option('--timelapse',
              prompt='Timelapse',
              is_flag=True,
              help='If images are timelapses.')
@click.option('--zstack',
              prompt='ZStack',
              is_flag=True,
              help='If images are zstacks.')
def main(dir_base, allslices=False, timelapse=False, zstack=False):
    if not isinstance(dir_base, str):
        raise TypeError(f'dir_base must be str but is {type(dir_base)}')
    if timelapse and zstack:
        raise ValueError('only select neither or one of timelapse/zstack, not both')
    if not os.path.exists(dir_base):
        raise ValueError('dir_base must exist')
    log.info(f'Started with dir_base "{dir_base}", timelapse "{timelapse}", zstack "{zstack}".')

    # Import
    try:
        dir_raw = os.path.join(dir_base, 'Raw')
        files = []
        for ext in EXTENSIONS:
            files.extend(glob.glob(f'{dir_raw}/*{ext}'))
        log.info(f'Found files - {files}.')
        images = list(map(adaptive_imread, files))
    except Exception:
        raise ValueError('Images could not be opened, check image format.')
    log.info(f'Images read successfully.')

    # Processing
    if timelapse:
        images = [process_timelapse(img) for img in images]
    elif zstack:
        images = [process_timelapse(img) for img in images]
    else:
        images = [process_standard(img) for img in images]
    log.info(f'Images processed.')

    # Saving
    for file, img in zip(files, images):
        basename = ntpath.basename(file).split('.')[0]

        if allslices:
            axis = get_min_axis(img)
            for i in range(min(img.shape)):
                curr_basename = f'{basename}_{i}'
                curr_img = np.take(img, i, axis=axis)
                save_image(dir_base, curr_basename, curr_img)
        else:
            save_image(dir_base, basename, img)

    print('\U0001F3C1 Programm finished successfully \U0001F603 \U0001F3C1')


if __name__ == "__main__":
    main()
