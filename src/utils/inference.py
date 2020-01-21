import click
import glob
import re
import numpy as np
import tensorflow as tf
import skimage
import matplotlib.pyplot as plt


def _normalize_image(input_image):
    ''' Normalizes image based on bit depth. '''
    assert type(input_image) == np.ndarray
    bit_depth = int(re.search(r'(\d+)', str(input_image.dtype)).group(0))
    bit_value = (2**bit_depth) - 1.
    return input_image.astype(np.float) / bit_value


@click.command()
@click.option('--model', help='Path to .h5 file.')
@click.option('--image_dir', help='Path to directory with images.')
def main(model, image_dir):
    assert model.endswith('.h5')

    model = tf.keras.models.load_model(model)

    print(image_dir)
    images = glob.glob(f"{image_dir}/*.tif")
    print(images)
    for image in images:
        pred_image = skimage.io.imread(image)
        pred_image = _normalize_image(pred_image)
        pred = model.predict(pred_image[None, ..., None])

        labels = ['Background', 'Foreground', 'Borders', 'Overlaps']
        channels = pred.shape[3]

        fig, ax = plt.subplots(1, channels+1, figsize=(20, 10))
        ax[0].imshow(pred_image)
        ax[0].set_title('Original Image')
        for i in range(channels):
            ax[i+1].imshow(pred[0, ..., i])
            ax[i+1].set_title(labels[i])
        plt.savefig(f'{image.split(".")[0]}.pdf')


if __name__ == "__main__":
    main()
