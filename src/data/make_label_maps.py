'''
Converts individual mask files into one label map.
The fourth and last step in the labeling workflow.
'''

import click
import glob
import logging
import numpy as np
import os
import skimage.io

LOG_FORMAT = '%(levelname)s %(asctime)s - %(filename)s %(funcName)s %(lineno)s - %(message)s'
logging.basicConfig(filename='./data.log',
                    level=logging.DEBUG,
                    format=LOG_FORMAT,
                    filemode='a')
log = logging.getLogger()


def import_masks(dir_masks):
    '''
    Imports masks in the format of FIJI labels (Unique files for each label as png).
    Converts them into one label mask with unique valued items.
    '''
    if not isinstance(dir_masks, str):
        raise TypeError(f'dir_masks must be str but is {type(dir_masks)}')
    if not os.path.exists(dir_masks):
        raise ValueError(f'dir_masks must exist, {dir_masks} does not')

    mask_files = glob.glob(f'{dir_masks}/mask_*.png')
    if not mask_files:
        raise ValueError(f'Empty masks directory. Must contain at least one item')
    masks = list(map(skimage.io.imread, mask_files))
    log.info(f'Found {len(masks)} masks.')

    masks = [np.where(m, i+1, 0) for i, m in enumerate(masks)]
    masks = np.max(masks, axis=0)
    return masks


@click.command()
@click.option('--base_dir',
              type=click.Path(exists=True),
              prompt='Path to the base directory',
              required=True,
              help='Path to the base directory.')
def main(base_dir):
    if not os.path.exists(base_dir):
        raise ValueError('base_dir must exist')
    log.info(f'Started with base_dir "{base_dir}".')

    dir_processed = os.path.join(base_dir, 'Processed')
    dir_labeling = os.path.join(base_dir, 'Labeling')
    dir_labeling_items = next(os.walk(dir_labeling))[1]
    log.info(f'Found labeling items - "{dir_labeling_items}".')

    for img_id in dir_labeling_items:

        dir_curr = os.path.join(dir_labeling, img_id)
        log.info(f'Current item - {dir_curr}.')

        # Images
        try:
            img = skimage.io.imread(f'{dir_curr}/images/{img_id}.tif')
        except Exception:
            raise ValueError(f'Could not read file {img_id}, please check file type.')
        skimage.io.imsave(f'{dir_processed}/images/{img_id}.tif',
                          img.astype(dtype=np.uint16),
                          check_contrast=False)
        log.info(f'Image processed.')

        # Masks
        mask = import_masks(f'{dir_curr}/masks')
        skimage.io.imsave(f'{dir_processed}/masks/{img_id}.tif',
                          mask.astype(dtype=np.uint16),
                          check_contrast=False)
        log.info(f'Mask processed.')

    print('\U0001F3C1 Programm finished successfully \U0001F603 \U0001F3C1')


if __name__ == "__main__":
    main()
