from flask import (
    Flask,
    render_template,
    request,
    make_response,
    send_from_directory,
    jsonify,
    redirect,
    url_for,
)
from werkzeug.utils import secure_filename

import logging
import numpy as np
import os
import scipy.ndimage as ndi
import skimage.color
import skimage.io
import skimage.measure
import skimage.morphology
import tensorflow as tf
import tifffile
import uuid

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["SECRET_KEY"] = str(uuid.uuid4())
app.config["MODEL_FOLDER"] = "static/models"
app.config["UPLOAD_FOLDER"] = "static/tmp_upload"
app.config["DISPLAY_FOLDER"] = "static/tmp_display"
app.config["PROCESSED_FOLDER"] = "static/tmp_processed"
app.config["INSTANCES_FOLDER"] = "static/tmp_instances"
app.jinja_env.filters["zip"] = zip

ALLOWED_EXTENSIONS = ["tif", "tiff", "jpg", "jpeg", "stk", "png"]
IMAGE_TYPES = ["One Frame (Grayscale or RGB)", "Z-Stack", "Time-Series"]
MODEL_IDS = {
    "Nuclear Semantic": "1XePQvBqgVx1zZZeYEryFd56ujgZumA9F",
    "Nuclear Instances": "166rnQYPQmzewIAjrbU7Ye-BhFotq2wnA",
    "Stress-Granules": "1SjjG4FbU9VkKTlN0Gvl7AsfiaoBKF7DB",
    "Cytoplasm (SunTag Background)": "1pVhyF81T4t7hh16dYKT3uJa5WCUQUI6X",
}
MODELS = {}
LOG_FORMAT = (
    "%(levelname)s %(asctime)s - %(filename)s %(funcName)s %(lineno)s - %(message)s"
)
logging.basicConfig(
    filename="./fluffy.log", level=logging.DEBUG, format=LOG_FORMAT, filemode="a"
)
log = logging.getLogger()


########################################
# Utils / helper functions
########################################


def load_model(model_id):
    """ Downloads and loads models into memory from google drive h5 files. """
    import gdown

    model_file = os.path.join(app.config["MODEL_FOLDER"], f"{model_id}.h5")
    if not os.path.exists(model_file):
        model_file = gdown.download(
            f"https://drive.google.com/uc?id={model_id}", model_file
        )
    model = tf.keras.models.load_model(model_file)
    return model


def adaptive_imread(file):
    """ Opens images depending on their filetype. """
    if not any(file.endswith(i) for i in ALLOWED_EXTENSIONS):
        raise ValueError(f"File must end with {ALLOWED_EXTENSIONS}")
    if any(
        file.endswith(i) for i in [".tif", ".tiff", ".jpg", ".jpeg", ".stk", ".png"]
    ):
        return skimage.io.imread(file)
    # if file.endswith('.czi'):
    #     return czifile.imread(file)


def adaptive_preprocessing(image, image_type):
    """ Preprocesses images according to their selected image type. """

    def __get_min_axis(image):
        """ Returns the index of a smallest axis of an image. """
        shape = image.shape
        axis = shape.index(min(shape))
        return axis

    image = image.squeeze()
    if not isinstance(image, np.ndarray):
        raise TypeError(f"image must be np.ndarray but is {type(image)}.")
    if not image.ndim in [2, 3]:
        raise ValueError(f"image must be 2D or 3D but is {image.ndim}D.")

    axis = __get_min_axis(image)
    axis_len = image.shape[axis]

    if image_type == "One Frame (Grayscale or RGB)":
        return [skimage.color.rgb2gray(image)]
    if image_type == "Z-Stack":
        return [np.max(image, axis=axis)]
    if image_type == "Time-Series":
        return [np.take(image, index, axis=axis)]
    if image_type == "All Frames":
        return [np.take(image, i, axis=axis) for i in range(axis_len)]


def predict_baseline(image, model, bit_depth=16):
    """
    Returns a binary or categorical model based prediction of an image.

    Args:
        - image (np.ndarray): Image to be predicted.
        - model (tf.keras.models.Model): Model used to predict the image.
        - bit_depth (int): Bit depth to normalize images. Model dependent.
    Returns:
        - pred (np.ndarray): Predicted image containing a probablilty map.
    """

    def __next_power(x, k=2):
        """ Calculates x's next higher power of k. """
        y, power = 0, 1
        while y < x:
            y = k ** power
            power += 1
        return y

    model_output = model.layers[-1].output_shape[-1]
    if model_output not in [1, 3]:
        raise ValueError(f"model_output must be 1 or 3 but is {model_output}")

    pred = image * (1.0 / (2 ** bit_depth - 1))
    pad_bottom = __next_power(pred.shape[0]) - pred.shape[0]
    pad_right = __next_power(pred.shape[1]) - pred.shape[1]
    pred = np.pad(pred, ((0, pad_bottom), (0, pad_right)), "reflect")
    pred = model.predict(pred[None, ..., None]).squeeze()
    pred = pred[: pred.shape[0] - pad_bottom, : pred.shape[1] - pad_right]
    return pred


def add_instances(pred_mask):
    """
    Adds instances to a categorical prediction with three layers (background, foreground, borders).

    Args:
        - pred_mask (np.ndarray): Prediction mask to for instances should be added.
    Returns:
        - pred (np.ndarray): 2D mask containing unique values for every instance.
    """
    if not isinstance(pred_mask, np.ndarray):
        raise TypeError(f"pred_mask must be np.ndarray but is {type(pred_mask)}")
    if not pred_mask.ndim == 3:
        raise ValueError(f"pred_mask must have 3 dimensions but has {pred_mask.ndim}")

    foreground_eroded = ndi.binary_erosion(pred_mask[..., 1] > 0.5, iterations=2)
    markers = skimage.measure.label(foreground_eroded)
    background = 1 - pred_mask[..., 0] > 0.5
    foreground = 1 - pred_mask[..., 1] > 0.5
    watershed = skimage.morphology.watershed(
        foreground, markers=markers, mask=background
    )

    mask_new = []
    for i in np.unique(watershed):
        mask_curr = watershed == i
        mask_curr = ndi.binary_erosion(mask_curr, iterations=2)
        mask_new.append(mask_curr * i)
    return np.argmax(mask_new, axis=0)


def adaptive_prediction(image, model, model_type):
    """
    Predicts images according to the selected model type.
    
    Args:
        - image (list of np.ndarray): List of images to be predicted.
        - model (tf.keras.models.Model): Model file.
    Returns:
        - image (np.ndarray): Array containing the prediction.
    """

    pred = []
    for i in image:
        curr_pred = predict_baseline(i, model)
        if "Instances" in model_type:
            curr_pred = add_instances(curr_pred)
        else:
            curr_pred = (curr_pred > 0.5) * 255
        pred.append(curr_pred.astype(np.uint8))
    return np.array(pred).squeeze()


def adaptive_imsave(fname, image, image_type):
    """ Saves images according to their selected image type. """

    def __save_colorize(fname, image):
        """ Colorizes images to better view labeled images. """
        import matplotlib.pyplot as plt

        cmap = plt.cm.jet
        norm = plt.Normalize(vmin=image.min(), vmax=image.max())
        image = cmap(norm(image))
        plt.imsave(fname, image)

    image = np.array(image).squeeze()

    # TODO name check image_type for IMAGE_TYPES and 'Instances
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Image must be np.ndarray but is {type(image)}.")
    if not image.ndim in [2, 3]:
        raise ValueError(f"Image must be 2D or 3D but is {image.ndim}D.")

    if image_type in ["One Frame (Grayscale or RGB)", "Z-Stack", "Time-Series"]:
        skimage.io.imsave(fname, image)
    if image_type == "All Frames":
        tifffile.imsave(fname, image, imagej=True)
    if image_type == "Instances":
        __save_colorize(fname, image)


def predict(file, image_type, model_type, single=False):
    """ Adaptively preprocesses, predicts, and saves images returning the filename(s). """
    log.info(
        f'Predicting with file "{file}", image "{image_type}", model {model_type}".'
    )

    # Naming
    fname = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(file.filename))
    ext = "png" if single else "tiff"
    basename = f'{secure_filename(file.filename).split(".")[0]}.{ext}'
    fname_in = os.path.join(app.config["DISPLAY_FOLDER"], basename)
    fname_out = os.path.join(app.config["PROCESSED_FOLDER"], basename)
    fname_instances = None
    file.save(fname)
    log.info(f'File "{fname}" saved.')

    # Processing
    original = adaptive_imread(file=fname)
    preprocessed = adaptive_preprocessing(image=original, image_type=image_type)
    prediction = adaptive_prediction(
        image=preprocessed, model=MODELS[model_type], model_type=model_type
    )
    log.info(f"Prediction returned.")

    # Saving
    if "Instances" in model_type:
        fname_instances = os.path.join(app.config["INSTANCES_FOLDER"], basename)
        adaptive_imsave(fname_instances, prediction, "Instances")
    if single:
        adaptive_imsave(fname=fname_in, image=preprocessed, image_type=image_type)
    adaptive_imsave(fname=fname_out, image=prediction, image_type=image_type)
    log.info(f"Predictions saved.")

    if single:
        return fname_in, fname_out, fname_instances
    return fname_out


########################################
# Setup / index
########################################


@app.before_first_request
def run_setup():
    """ Setup before responding to requests. Downloads all models. """
    global MODELS
    for name, model_id in MODEL_IDS.items():
        MODELS[name] = load_model(model_id)
        log.info(f'Loaded model "{name}".')
    log.info("Model loading complete.")


@app.route("/")
@app.route("/index")
def index():
    return render_template("index.html", landing=True)


########################################
# Single predictions
########################################


@app.route("/single")
def single():
    """ Standard "Single Images" page. """
    image_selection = request.cookies.get("image_selection")
    model_selection = request.cookies.get("model_selection")

    return render_template(
        "single.html",
        title="Single Images",
        image_options=IMAGE_TYPES,
        image_selection=image_selection,
        model_options=list(MODEL_IDS.keys()),
        model_selection=model_selection,
    )


@app.route("/predict_single", methods=["POST"])
def predict_single():
    """ API to predict on multiple image files, returning the location of input and predicted images. """
    file = request.files["file"]
    image_selection = request.form.get("image")
    model_selection = request.form.get("model")
    log.info(
        f"Single selections - file: {file}, image: {image_selection}, model: {model_selection}."
    )

    fname_in, fname_out, fname_instances = predict(
        file=file, image_type=image_selection, model_type=model_selection, single=True
    )
    if fname_instances is None:
        pred_display = fname_out
    else:
        pred_display = fname_instances

    resp = make_response(
        render_template(
            "prediction.html",
            title="Prediction",
            original=fname_in,
            prediction=fname_out,
            prediction_display=pred_display,
            # import itertools
            # zipped_list=list(itertools.chain(*list(zip(fname_in, fname_out))))
        )
    )
    resp.set_cookie("image_selection", image_selection)
    resp.set_cookie("model_selection", model_selection)
    return resp


########################################
# Batch predictions
########################################


@app.route("/batch")
def batch():
    """ Standard "Batch Processing" page. """
    image_selection = request.cookies.get("image_selection")
    model_selection = request.cookies.get("model_selection")
    image_types = IMAGE_TYPES.copy()
    image_types.append("All Frames")

    return render_template(
        "batch.html",
        title="Batch Prediction",
        image_options=image_types,
        image_selection=image_selection,
        model_options=list(MODEL_IDS.keys()),
        model_selection=model_selection,
    )


@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    """ API to predict on multiple image files, returning the location of predicted images. """
    uuids = request.form.getlist("uuid")
    files = request.files.getlist("file")
    image_selection = request.form.get("image")
    model_selection = request.form.get("model")
    log.info(
        f"Batch selections - ids: {uuids}, files: {files}, image: {image_selection}, model: {model_selection}."
    )

    response = {
        uuid: predict(file, image_selection, model_selection)
        for uuid, file in zip(uuids, files)
    }
    return jsonify(response)


########################################
# Help
########################################


@app.route("/help")
def help():
    return render_template("help.html", title="Help")


########################################
# Utils / helper routes
########################################


@app.route("/download/<path:filename>")
def download(filename):
    """ API to download a single filename. """
    filename = filename.split("/")[-1]
    return send_from_directory(
        app.config["PROCESSED_FOLDER"], filename, as_attachment=True
    )


@app.route("/delete/<path:filename>/")
def delete(filename):
    """ Deletes a single filename from all temporary image folders. """
    for path in [
        "UPLOAD_FOLDER",
        "DISPLAY_FOLDER",
        "PROCESSED_FOLDER",
        "INSTANCES_FOLDER",
    ]:
        for ext in ALLOWED_EXTENSIONS:
            name = os.path.join(app.config[path], f'{filename.split(".")[0]}.{ext}')
            if os.path.exists(name):
                os.remove(name)
    return "success"


@app.route("/delete_all/<url>")
def delete_all(url):
    """ Deletes all files stored in temporary image folders. """

    import glob

    for path in [
        "UPLOAD_FOLDER",
        "DISPLAY_FOLDER",
        "PROCESSED_FOLDER",
        "INSTANCES_FOLDER",
    ]:
        files = glob.glob(f"{app.config[path]}/*")
        for f in files:
            os.remove(f)
    log.info("All files removed.")

    return redirect(url_for(url))


if __name__ == "__main__":
    app.run(debug=True)
