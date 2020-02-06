import click
import cv2
import glob
import numpy as np
import os
import pathlib
import scipy.ndimage as ndi
import skimage.io
import tensorflow as tf


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
    if not bit_depth == 0:
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

    foreground_eroded = ndi.morphology.binary_erosion(pred_mask[..., 1] > 0.5, iterations=2)
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
@click.option('--model_file', type=str, required=True, help='Location of the h5 file with the model.')
@click.option('--image_file', type=str, default=None, help='Name of model.')
@click.option('--image_folder', type=str, default=None, help='Name of model.')
def main(model_file, image_file, image_folder):
    if not model_file.endswith('.h5'):
        raise ValueError('Model file not of type h5')
    if (not image_file and not image_folder) or (image_file and image_folder):
        raise ValueError('Must define either file or folder')

    # File checking
    if image_folder:
        extensions = ['png', 'jpg', 'jpeg', 'stk', 'tif', 'tiff']
        file_names = sorted(glob.glob(f'{image_folder}/*.{extensions}'))
    if image_file:
        file_names = glob.glob(image_file)
    if not file_names:
        raise ValueError(f'Folder empty or no files of supported types: {extensions}')

    base_names = [pathlib.Path(f).stem for f in file_names]
    base_folder = os.path.dirname(file_names[0])
    os.makedirs(f'{base_folder}/predictions/', exist_ok=True)

    # Model import
    try:
        model = tf.keras.models.load_model(model_file)
    except Exception:
        raise TypeError(f'Model is not of type tf.keras.models')

    # Predictions
    for i, file in enumerate(file_names):
        try:
            curr_img = skimage.io.imread(file)
        except Exception:
            raise ValueError('Image failed to open check for proper formatting')
        if curr_img.ndim != 2:
            raise ValueError(f'Image must have 2 dimensions but has {curr_img.ndim}')
        curr_pred = predict(curr_img, model)
        skimage.io.imsave(f'{base_folder}/predictions/{base_names[i]}.png', curr_pred)


if __name__ == "__main__":
    main()
