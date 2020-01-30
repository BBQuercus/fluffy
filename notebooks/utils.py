import cv2
import numpy as np
import tensorflow as tf
import scipy.ndimage as ndi
import skimage.filters
import skimage.feature
import skimage.segmentation

model_binary = None
model_categorical = None


def _next_power(x, k=2):
    y, power = 0, 1
    while y < x:
        y = k**power
        power += 1
    return y


def predict_otsu(image, add_instances=False):
    '''
    '''
    thresh_otsu = skimage.filters.threshold_otsu(image)
    pred = image > thresh_otsu

    if not add_instances:
        return pred.astype(dtype=np.uint8)

    distance = ndi.distance_transform_edt(pred)
    local_maxi = skimage.feature.peak_local_max(pred, indices=False, footprint=np.ones((3, 3)), labels=image)
    markers = ndi.label(local_maxi)[0]
    seg = skimage.segmentation.watershed(-distance, markers, mask=pred, connectivity=0, watershed_line=True)

    return seg


def predict_binary(image, model, bit_depth=16):
    '''
    '''
    pred = image * (1./(2**bit_depth - 1))
    pad_bottom = _next_power(pred.shape[0]) - pred.shape[0]
    pad_right = _next_power(pred.shape[1]) - pred.shape[1]
    pred = cv2.copyMakeBorder(pred, 0, pad_bottom, 0, pad_right, cv2.BORDER_REFLECT)
    pred = model.predict(pred[None, ..., None]).squeeze()
    pred = pred[:pred.shape[0]-pad_bottom, :pred.shape[1]-pad_right]
    return pred


def predict_categorical(image, model, add_instances=False, bit_depth=16):
    '''
    '''
    # Prediction
    pred = image * (1./(2**bit_depth - 1))
    pad_bottom = _next_power(pred.shape[0]) - pred.shape[0]
    pad_right = _next_power(pred.shape[1]) - pred.shape[1]
    pred = cv2.copyMakeBorder(pred, 0, pad_bottom, 0, pad_right, cv2.BORDER_REFLECT)
    pred = model.predict(pred[None, ..., None]).squeeze()
    pred = pred[:pred.shape[0]-pad_bottom, :pred.shape[1]-pad_right, :]

    if not add_instances:
        return pred[..., 1]

    # Instance separation
    foreground_ero = ndi.morphology.binary_erosion(pred[..., 1] > 0.5, iterations=2)
    markers = skimage.measure.label(foreground_ero)
    mask = 1-pred[..., 0] > 0.5
    curr_img = 1-pred[..., 1] > 0.5
    water = skimage.morphology.watershed(curr_img, markers=markers, mask=mask)
    mask_new = []
    for i in np.unique(water):
        mask_curr = water==i
        mask_curr = ndi.binary_erosion(mask_curr, iterations=2)
        mask_new.append(mask_curr * i)
    pred = np.argmax(mask_new, axis=0)

    return pred


def evaluate_accuracy(y_true, y_pred):
    y_true = y_true > 0
    m = tf.keras.metrics.BinaryAccuracy()
    m.update_state(y_true, y_pred)
    return m.result().numpy()


def evaluate_auc(y_true, y_pred):
    y_true = y_true > 0
    m = tf.keras.metrics.AUC(num_thresholds=3)
    m.update_state(y_true, y_pred)
    return m.result().numpy()


def evaluate_mse(y_true, y_pred):
    y_true = y_true > 0
    m = tf.keras.metrics.MeanSquaredError()
    m.update_state(y_true, y_pred)
    return m.result().numpy()


def evaluate_iou(y_true, y_pred, threshold=0.5):
    y_true = y_true > 0
    y_pred = y_pred > threshold
    m = tf.keras.metrics.MeanIoU(num_classes=2)
    m.update_state(y_true, y_pred)
    iou = m.result().numpy()
    return round(iou, ndigits=4)


def evaluate_all(y_true, y_pred):
    metrics = [evaluate_accuracy, evaluate_auc, evaluate_mse, evaluate_iou]
    metrics_name = ['Accuracy', 'AUC', 'MSE', 'IOU']

    for metric, name in zip(metrics, metrics_name):
        print(f'{name}: {metric(y_true, y_pred)}')


def compare_methods(image, mask, model_binary, model_categorical):
    names = ['Otsu', 'Binary', 'Categorical']
    predictions = [predict_otsu(image),
                   predict_binary(image, model_binary),
                   predict_categorical(image, model_categorical)]

    for name, prediction in zip(names, predictions):
        print('–––––––––––')
        print(name)
        evaluate_all(mask, prediction)
