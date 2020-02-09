'''
Populates a directory substructure at a specified location.
The first step in the labeling workflow.
'''

import click
import logging
import os

LOG_FORMAT = '%(levelname)s %(asctime)s - %(filename)s %(funcName)s %(lineno)s - %(message)s'
logging.basicConfig(filename='./data.log',
                    level=logging.DEBUG,
                    format=LOG_FORMAT,
                    filemode='a')
log = logging.getLogger()


@click.command()
@click.option('--base_dir',
              type=click.Path(exists=True),
              prompt='Path to the base directory',
              required=True,
              help='Path to the base directory.')
@click.option('--name',
              type=str,
              prompt='Folder name',
              required=True,
              help='Name of the to be generated folder.')
def main(base_dir, name):
    if not isinstance(base_dir, str):
        raise TypeError(f'base_dir must be str but is {type(base_dir)}')
    if not isinstance(name, str):
        raise TypeError(f'name must be str but is {type(name)}')
    if not os.path.exists(base_dir):
        raise ValueError('base_dir must exist')
    log.info(f'Started with base_dir "{base_dir}"  and name "{name}".')

    # Root
    root = os.path.join(base_dir, name)
    os.mkdir(root)
    log.info('Created root.')

    # Raw
    root_raw = os.path.join(root, 'Raw')
    os.mkdir(root_raw)
    log.info('Created Raw.')

    # Labeling
    root_labeling = os.path.join(root, 'Labeling')
    os.mkdir(root_labeling)
    log.info('Created Labeling.')

    # Processed
    root_processed = os.path.join(root, 'Processed')
    os.mkdir(root_processed)
    for subdir in ['images', 'masks']:
        root_subdir = os.path.join(root_processed, subdir)
        os.mkdir(root_subdir)
    log.info('Created Processed.')

    print('\U0001F3C1 Programm finished successfully \U0001F603 \U0001F3C1')


if __name__ == "__main__":
    main()
