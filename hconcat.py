import cv2
import os

def modif_filename(filename, motif="_concat"):
    basename, ext = os.path.splitext(filename)
    return f'{basename}{motif}{ext}'

def concat_images(dir1, dir2, dir_out):
    images = [f for f in os.listdir(dir1) if os.path.isfile(os.path.join(dir1, f))]
    for f in images:
        im1 = cv2.imread(dir1 + "/" + f)
        im2 = cv2.imread(dir2 + "/" + f)
        im_h = cv2.hconcat([im1, im2])
        result_name = modif_filename(f)
        cv2.imwrite(dir_out + "/" + result_name, im_h)
        print(f'{result_name} written')

DIR1 = 'plign_masked'
DIR2 = 'plign_masked2'
DIR_OUT = 'concatpl'

concat_images(DIR1, DIR2, DIR_OUT)