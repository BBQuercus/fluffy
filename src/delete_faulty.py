import os
import click
import ntpath
import glob
import skimage.io


@click.command()
@click.option('--path_in', help='Path of files to be changed.')
# @click.option('--path_out', help='Path where files will be saved.')
@click.option('--ext', help='Extensions of files to be changed.')
def main(path_in, ext):
    images = sorted(glob.glob(f'{path_in}/images/*.{ext}'))
    masks = sorted(glob.glob(f'{path_in}/masks/*.{ext}'))

    for f_image, f_mask in zip(images, masks):
        image = skimage.io.imread(f_image)
        mask = skimage.io.imread(f_mask)

        if image.shape[:2] == mask.shape[:2]:
            continue
        else:
            os.remove(f_image)
            os.remove(f_mask)
            print(f'Removed: {ntpath.basename(f_image)}')
            print('––––––––')
            # min_dim_0 = min(image.shape[0], mask.shape[0])
            # min_dim_1 = min(image.shape[1], mask.shape[1])
            # image = image[:min_dim_0, :min_dim_1]
            # mask = mask[:min_dim_0, :min_dim_1]

            # skimage.io.imsave(f'{path_out}/images/{ntpath.basename(f_image)}', image)
            # skimage.io.imsave(f'{path_out}/masks/{ntpath.basename(f_mask)}', mask)


if __name__ == "__main__":
    main()
