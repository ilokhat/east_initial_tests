import os
import matplotlib.pyplot as plt

from skimage import color, io, img_as_uint
from skimage.morphology import closing, opening
from skimage.filters import try_all_threshold
from skimage.filters import (threshold_otsu, threshold_niblack,threshold_sauvola)

threshold = {'otsu': threshold_otsu, 'niblack': threshold_niblack, 'sauvola': threshold_sauvola }
def thresh_close(img, t='otsu'):
    f = threshold[t]
    thresh = f(img)
    bw = closing(img > thresh)
    return bw

def preprocess_all_images_in(images_dir, result_dir, t='otsu'):
    onlyfiles = [f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))]
    for f in onlyfiles:
        image_name = images_dir + "/" + f
        basename, ext = os.path.splitext(f)
        result_name = f'{result_dir}/{basename}_{t}_{ext}'
        img = io.imread(image_name, as_gray=True)
        img_new = thresh_close(img, t)
        io.imsave(result_name, img_as_uint(img_new))
        print(image_name , "->", result_name)


IMG_DIR = './res2'
IMG_RES = f'./prepro2'

IMG_DIR = '../wms_grid_gen/tiles'
IMG_RES = '../wms_grid_gen/tiles_prepro'


#preprocess_all_images_in(IMG_DIR, IMG_RES, t=THR)
for th in threshold:
    print(60*'*')
    #IMG_RES = f'./prepro_{th}'
    preprocess_all_images_in(IMG_DIR, IMG_RES, t=th)
    print(60*'*')

# rad = 'img_6_3'
# imagefile = f'res/{rad}.jpg'
# img = io.imread(imagefile, as_gray=True)
# # img = io.imread(imagefile)
# # img = color.rgb2lab(img)
# # img[:,:,1] = img[:,:,2] = 0
# # img = color.lab2rgb(img)
# # img = img[:,:,0]

# thresh = threshold_otsu(img)
# bw = closing(img > thresh)
# #bw = opening(img < thresh)
# plt.imshow(bw, cmap='gray')
# plt.show()

# # th = threshold_otsu(img)
# # img[img <= th] = 0
# # img[img > th] = 1
# # img = 1 - img
# # plt.imshow(img, cmap='gray')
# # plt.show()

# # # remove artifacts connected to image border
# # #cleared = clear_border(bw)

# # #fig, ax = try_all_threshold(img, figsize=(10, 8), verbose=False)
# # #plt.show()

# # #img2 = closing(img)
# #io.imsave(f'{rad}_bin3.jpg', img_as_uint(img))
