import click
import glob
import numpy as np
import skimage.io
import skimage.measure


# Format:
# dirpath xmin,ymin,w,h,c...
# Example:
# dirpath 122,187,136,201,7 71,209,85,223,4


@click.command()
@click.option('--path', help='Directory of files / masks.')
@click.option('--ext', default='tif', help='Directory of files / masks.')
def main(path, ext):
    files = glob.glob(f'{path}/*{ext}')
    output = []
    for f in files:
        img = skimage.io.imread(f)
        labels = skimage.measure.regionprops(img)

        # Format (min_row, min_col, max_row, max_col)
        bboxes = [l.bbox for l in labels]

        # Format xywhc
        bboxes = [[str(i[0]),
                   str(i[1]),
                   str(i[2]),
                   str(i[3]),
                   str(1)]
                  for i in bboxes]

        bboxes = ' '.join(','.join(bbox) for bbox in bboxes)
        output.append(' '.join([f, bboxes]))

    with open(f'{path}/bboxes.txt', 'w') as f:
        for out in output:
            f.write(f'{out}\n')


if __name__ == '__main__':
    main()
