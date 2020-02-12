import click
import cv2
import glob
import logging
import numpy as np
import os
import pathlib
import scipy.ndimage as ndi
import skimage.io
import tensorflow as tf

EXTENSIONS = ['.png', '.jpg', '.jpeg', '.stk', '.tif', '.tiff']
LOG_FORMAT = '%(levelname)s %(asctime)s - %(filename)s %(funcName)s %(lineno)s - %(message)s'
logging.basicConfig(filename='./model.log',
                    level=logging.DEBUG,
                    format=LOG_FORMAT,
                    filemode='a')
log = logging.getLogger()


def _next_power(x, k=2):
    ''' Calculates x's next higher power of k. '''
    y, power = 0, 1
    while y < x:
        y = k**power
        power += 1
    return y


def predict(image, model, add_instances=False, bit_depth=16):
    '''
    Returns an binary or categorical model based prediction of an image.

    Args:
        - image (np.ndarray): Image to be predicted.
        - model (tf.keras.models.Model): Model used to predict the image.
        - add_instances (bool): Optional separation of instances.
    Returns:
        - pred (np.ndarray): One of two predicted images:
            - If add_instances is False the unthresholded foreground.
            - If add_instances is True thresholded instances.
    '''
    if not isinstance(image, np.ndarray):
        raise TypeError(f'')
    if not isinstance(model, tf.keras.models.Model):
        raise TypeError(f'')
    if not isinstance(add_instances, bool):
        raise TypeError(f'')
    if not isinstance(bit_depth, int):
        raise TypeError(f'')
    if bit_depth == 0:
        raise ZeroDivisionError(f'bit_depth must not be zero')

    model_output = model.layers[-1].output_shape[-1]
    if model_output not in [1, 3]:
        raise ValueError(f'model_output not in 1, 3 but is {model_output}')

    pred = image * (1./(2**bit_depth - 1))
    pad_bottom = _next_power(pred.shape[0]) - pred.shape[0]
    pad_right = _next_power(pred.shape[1]) - pred.shape[1]
    pred = cv2.copyMakeBorder(pred, 0, pad_bottom, 0, pad_right, cv2.BORDER_REFLECT)
    pred = model.predict(pred[None, ..., None]).squeeze()

    if model_output == 1:
        pred = pred[:pred.shape[0]-pad_bottom, :pred.shape[1]-pad_right]
        return pred
    if not add_instances:
        pred = pred[:pred.shape[0]-pad_bottom, :pred.shape[1]-pad_right, :]
        return pred[..., 1]
    pred = add_instances(pred)
    return pred


def add_instances(pred_mask):
    '''
    Adds instances to a categorical prediction with three layers (background, foreground, borders).

    Args:
        - pred_mask (np.ndarray): Prediction mask to for instances should be added.
    Returns:
        - pred (np.ndarray): 2D mask containing unique values for every instance.
    '''
    if not isinstance(pred_mask, np.ndarray):
        raise TypeError(f'pred_mask must be np.ndarray but is {type(pred_mask)}')
    if not pred_mask.ndim == 3:
        raise ValueError(f'pred_mask must have 3 dimensions but has {pred_mask.ndim}')

    foreground_eroded = ndi.binary_erosion(pred_mask[..., 1] > 0.5, iterations=2)
    markers = skimage.measure.label(foreground_eroded)
    background = 1-pred_mask[..., 0] > 0.5
    foreground = 1-pred_mask[..., 1] > 0.5
    watershed = skimage.morphology.watershed(foreground, markers=markers, mask=background)

    mask_new = []
    for i in np.unique(watershed):
        mask_curr = watershed == i
        mask_curr = ndi.binary_erosion(mask_curr, iterations=2)
        mask_new.append(mask_curr * i)
    return np.argmax(mask_new, axis=0)


@click.command()
@click.option('--model_file',
              type=str,
              prompt='Model file',
              required=True,
              help='Location of the h5 file with the model.')
@click.option('--image',
              type=click.Path(exists=True),
              prompt='Image file/folder',
              required=True,
              help='Location of image file or folder.')
def main(model_file, image):
    if not all(isinstance(i, str) for i in [model_file, image]):
        raise TypeError(f'model_file, image must be str but are {type(model_file), type(image)}.')
    if not model_file.endswith('.h5'):
        raise ValueError('model_file not of type h5.')
    if not os.path.exists(image):
        raise ValueError(f'image file/folder must exist. {image} does not.')
    if (not os.path.isdir(image)) and (not any(image.endswith(i) for i in EXTENSIONS)):
        raise ValueError(f'image file must be of type {EXTENSIONS}.')
    log.info(f'Started with base_dir "{model_file}" and name "{image}".')

    # File checking
    if os.path.isdir(image):
        file_names = sorted(glob.glob(f'{image}/*.{EXTENSIONS}'))
    else:
        file_names = glob.glob(image)
    if not file_names:
        raise ValueError(f'Folder empty or no files of supported types: {EXTENSIONS}')
    log.info(f'Found files - {file_names}.')

    base_names = [pathlib.Path(f).stem for f in file_names]
    dir_base = os.path.dirname(file_names[0])
    dir_prediction = os.path.join(dir_base, 'predictions')
    os.makedirs(dir_prediction, exist_ok=True)
    log.info('Created prediction.')

    # Model import
    try:
        model = tf.keras.models.load_model(model_file)
    except Exception:
        raise TypeError(f'Model is not of type tf.keras.models')
    log.info('Model loaded successfully.')

    # Predictions
    for i, file in enumerate(file_names):
        try:
            curr_img = skimage.io.imread(file)
        except Exception:
            raise ValueError('Image failed to open check for proper formatting')
        if curr_img.ndim != 2:
            raise ValueError(f'Image must be 2D but is {curr_img.ndim}D')
        log.info(f'Image {file} loaded.')
        curr_pred = predict(curr_img, model)
        skimage.io.imsave(f'{dir_prediction}/{base_names[i]}.png', curr_pred)
        log.info('Prediction saved.')

    print('\U0001F3C1 Programm finished successfully \U0001F603 \U0001F3C1')


if __name__ == "__main__":
    main()
