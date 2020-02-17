import cv2
import gdown
import glob
import numpy as np
import os
import re
import skimage.io
import skimage.measure
import skimage.morphology
import skimage.color
import streamlit as st
import scipy.ndimage as ndi
import tensorflow as tf
from matplotlib import cm

EXTENSIONS = ['.png', '.jpg', '.jpeg', '.stk', '.tif', '.tiff']
MODEL_IDS = {
    'None': None,
    'Nuclear Semantic': '1XePQvBqgVx1zZZeYEryFd56ujgZumA9F',
    'Nuclear Instances': '166rnQYPQmzewIAjrbU7Ye-BhFotq2wnA',
    'Stress-Granules': '1SjjG4FbU9VkKTlN0Gvl7AsfiaoBKF7DB',
    'Cytoplasm (SunTag Background)': '1pVhyF81T4t7hh16dYKT3uJa5WCUQUI6X',
    'P-Bodies': None,
    'Spots': None
}

s1_checkpoint = True
s2_checkpoint = False
s3_checkpoint = False
s4_checkpoint = False
s5_checkpoint = False
s6_checkpoint = False


def adaptive_import(file):
    ''' Imports image files based on their filetype. '''
    if file.endswith('.jpg'):
        image = cv2.imread(file).squeeze()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # elif s1_file.endswith('.czi'):
    #     image = czi.imread(s1_file)
    else:
        image = skimage.io.imread(file).squeeze()
    return image


def get_min_axis(image):
    ''' Returns the index of a smallest axis of an image. '''
    shape = image.shape
    axis = shape.index(min(shape))
    return axis


def adaptive_preprocessing(image, image_type):
    ''' Preprocesses image according to selected image type. '''

    image = image.squeeze()
    axis = get_min_axis(image)
    axis_len = image.shape[axis]

    if image_type == 'Grayscale':
        return [image]
    if image_type == 'RGB':
        return [skimage.color.rgb2gray(image)]
    if image_type == 'Z-Stack':
        return [np.max(image, axis=axis)]
    if image_type == 'Time-Lapse':
        index = axis_len // 2
        return [np.take(image, index, axis=axis)]
    if image_type == 'All slices':
        return [np.take(image, i, axis=axis) for i in range(axis_len)]


def normalize(image):
    ''' Normalizes image for better displaying. '''
    uint = re.search(f'[0-9]+', str(image.dtype))[0]
    bits = 2**int(uint)-1
    curr_image = image / bits
    return curr_image / curr_image.max()


def adaptive_display(image, prediction=False):
    ''' Displays preprocessed image accordingly. '''

    if not image:
        return None
    if len(image) == 1:
        if prediction:
            image_display = cm.jet(image[0])
        else:
            image_display = normalize(image[0])
        st.image(image_display, caption='Your very fluffy image', use_column_width=True)
    else:
        slider = st.slider('Index', 1, len(image)-1)
        image_display = normalize(image[slider])
        st.image(image_display, caption='Your very fluffy image', use_column_width=True)


def _next_power(x, k=2):
    ''' Calculates x's next higher power of k. '''
    y, power = 0, 1
    while y < x:
        y = k**power
        power += 1
    return y


def predict_baseline(image, model, bit_depth=16):
    '''
    Returns a binary or categorical model based prediction of an image.

    Args:
        - image (np.ndarray): Image to be predicted.
        - model (tf.keras.models.Model): Model used to predict the image.
        - categorical (bool): Optional separation of instances.
            Only applies if model provides categorical output.
    Returns:
        - pred (np.ndarray): One of two predicted images:
            - If add_instances is False the unthresholded foreground.
            - If add_instances is True thresholded instances.
    '''

    model_output = model.layers[-1].output_shape[-1]
    if model_output not in [1, 3]:
        raise ValueError(f'model_output not in 1, 3 but is {model_output}')

    pred = image * (1./(2**bit_depth - 1))
    pad_bottom = _next_power(pred.shape[0]) - pred.shape[0]
    pad_right = _next_power(pred.shape[1]) - pred.shape[1]
    pred = cv2.copyMakeBorder(pred, 0, pad_bottom, 0, pad_right, cv2.BORDER_REFLECT)
    pred = model.predict(pred[None, ..., None]).squeeze()
    pred = pred[:pred.shape[0]-pad_bottom, :pred.shape[1]-pad_right]
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


def adaptive_prediction(image, model, model_type, asarray=False):

    pred = []
    for n, i in enumerate(image):
        if model_type == 'None':
            st.error('Please select a model type.')
        elif model_type == 'Nuclear Semantic':
            pred.append(predict_nuclear_semantic(i, model))
        elif model_type == 'Nuclear Instances':
            pred.append(predict_nuclear_instances(i, model))
        elif model_type == 'Stress-Granules':
            pred.append(predict_stress_granules(i, model))
        elif model_type == 'Cytoplasm (SunTag Background)':
            pred.append(predict_cytoplasm(i, model))

    if asarray:
        pred = np.array(pred).squeeze()
    return pred


def predict_nuclear_instances(image, model):
    pred = predict_baseline(image, model)
    pred = add_instances(pred)
    return pred.astype(np.uint8)


def predict_nuclear_semantic(image, model):
    pred = predict_baseline(image, model)
    pred = (pred > 0.5) * 255
    return pred.astype(np.uint8)


def predict_stress_granules(image, model):
    pred = predict_baseline(image, model)
    pred = (pred > 0.5) * 255
    return pred.astype(np.uint8)


def predict_cytoplasm(image, model):
    pred = predict_baseline(image, model)
    pred = (pred > 0.5) * 255
    return pred.astype(np.uint8)


fluffy = '../../data/fluffy.jpg'
st.image(fluffy, use_column_width=True)
st.title('Fluffy')
st.write('''Hello there! – This is the fluffiest image processing you will have ever done.
Simply follow the steps and you'll be fluffed in no time.''')


# Step 1
if s1_checkpoint:
    s1_header = st.subheader('1. Import a fluffy image')

    s1_file = st.text_input('File path')
    if s1_file:
        if not os.path.exists(s1_file):
            st.error(f'The file must exist.')
        elif not any(s1_file.endswith(i) for i in EXTENSIONS):
            st.error(f'The file must end in {EXTENSIONS}.')
        else:
            image = adaptive_import(s1_file)
            s2_checkpoint = True
            st.success(f'File read successfully.')

    s1_info = st.selectbox('\U0001F4A1 How do I get the file path on:', options=['Operating System', 'Mac', 'Windows'])
    if s1_info == 'Mac':
        st.write('''
            - Right click on the desired file.
            - Press and hold the option (alt) key.
            - Select "Copy "FILENAME" as Pathname".
            - Paste into the input box above and click enter.
            - Alternatively drag the file into a terminal window. This displays the full path.
            ''')
    if s1_info == 'Windows':
        st.write('''
            - Press and hold shift and right click on the desired file.
            - Select "Copy as Path".
            - Paste into the input box above and click enter.
            - Alternatively in the file manager select "Copy path" in the "Home" tab under "Clipboard".
            ''')


# Step 2
if s2_checkpoint:
    s2_header = st.subheader('2. Assign a fluffy image type')

    if image.ndim == 2:
        st.info('Grayscale image detected. No need to assign the image type.')
        image_type = 'Grayscale'
        s3_checkpoint = True
    elif image.ndim == 3 and min(image.shape) == 3:
        st.info('RGB image detected. No need to assign the image type.')
        image_type = 'RGB'
        s3_checkpoint = True
    elif image.ndim == 3:
        image_type = st.selectbox('Image type', ['None', 'All slices', 'Z-Stack', 'Time-Lapse'])
        s1_info = st.selectbox('\U0001F4A1 What does image type mean?', options=['Option', 'All slices', 'Z-Stack', 'Time-Lapse'])
        if s1_info == 'All slices':
            st.write('''
                - Predictions will be performed on every slice / frame.
                - This will take substantially longer.
                - Use this only when dealing with very uneven or moving objects.
                ''')
        if s1_info == 'Z-Stack':
            st.write('''
                - A maxium projection will be performed.
                - Use this option when dealing with evenly dispersed objects in the Z axis.
                ''')
        if s1_info == 'Time-Lapse':
            st.write('''
                - A single frame will be selected for prediction.
                - Use this option when objects don't move alot.
                ''')
    else:
        st.error('Image is strangely shaped. Make sure it is 2D or 3D.')

    if not image_type == 'None':
        image = adaptive_preprocessing(image, image_type)
        if st.checkbox('View image?'):
            adaptive_display(image)
        s3_checkpoint = True


# Step 3
if s3_checkpoint:
    s3_header = st.subheader('3. Select a fluffy model')

    # output_type = st.radio('Raw model output - manual probability cutoff required', ['No', 'Yes'])
    model_type = st.selectbox('Model type', list(MODEL_IDS.keys()))

    if not model_type == 'None':
        model_id = MODEL_IDS[model_type]
        model_file = f'./tmp/{model_id}.h5'
        if not os.path.exists(model_file):
            os.makedirs('./tmp', exist_ok=True)
            model_file = gdown.download(f'https://drive.google.com/uc?id={model_id}', model_file)
        model = tf.keras.models.load_model(model_file)
        st.success('Model loaded successfully.')
        s4_checkpoint = True


# Step 4
if s4_checkpoint:
    st.subheader('4. Preview the fluffy model')

    pred = adaptive_prediction(image, model, model_type)
    adaptive_display(pred, prediction=True)

    s5_checkpoint = True


# Step 5
if s5_checkpoint:
    st.subheader('5. Batch processing')
    dir_in = st.text_input('Folder to process', value=os.path.dirname(s1_file))
    regex = st.text_input('Unique Identifier')
    dir_out = st.text_input('Output folder', value=os.path.join(os.path.dirname(s1_file), 'tmp'))
    if dir_in and not os.path.exists(dir_in):
        st.error('Your input does not exist. Try again.')
        s5_success = False
    if dir_in and os.path.exists(dir_in):
        st.success('Your input was found.')
        s5_success = True

        files = []
        for ext in EXTENSIONS:
            files.extend(glob.glob(f'{dir_in}/*{regex}*{ext}'))
        basenames = [os.path.basename(i) for i in files]
        st.write('The following files were found:')
        st.json(basenames)

    if s5_success and st.button('Run Batch'):
        s5_bar = st.progress(0)

        os.makedirs(dir_out, exist_ok=True)
        for n, file in enumerate(files):
            curr_image = adaptive_import(file)
            curr_image = adaptive_preprocessing(curr_image, image_type)
            curr_pred = adaptive_prediction(curr_image, model, model_type, True)
            curr_name = f'{dir_out}/{basenames[n].split(".")[0]}.tif'
            skimage.io.imsave(curr_name, curr_pred)
            s5_bar.progress(n // len(files) * 100)

        s5_bar.progress(100)
        s6_checkpoint = True


# Step 6
if s6_checkpoint:
    st.subheader('6. You\'re done for now \U0001F60A')

# About
st.subheader('About')
st.markdown('''
    If you find this helpful for your research please cite:
    ```
    @misc{Fluffy,
        author = {Bastian Th., Eichenberger},
        title = {Fluffy},
        year = {2020},
        publisher = {GitHub},
        journal = {GitHub repository},
        howpublished = {\\url{https://github.com/bbquercus/fluffy}},
        commit = {4f57d6a0e4c030202a07a60bc1bb1ed1544bf679}}
    ```
    For assistance or to report bugs, please raise an issue on [GitHub](https://github.com/bbquercus/fluffy/issues).

    Image designed by vectorpocket / [Freepik](http://www.freepik.com).
    ''')
