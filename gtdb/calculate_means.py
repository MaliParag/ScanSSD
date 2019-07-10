"""
Author: Parag Mali
Calculate the image per channel mean and standard
deviation in the training set, do not calculate the statistics on the
whole dataset, as per here http://cs231n.github.io/neural-networks-2/#datapre
"""

import numpy as np
from os import listdir
from os.path import join, isdir
from glob import glob
import cv2
import timeit
import sys
from multiprocessing import Pool
sys.path.extend(['/home/psm2208/code', '/home/psm2208/code'])

# number of channels of the dataset image, 3 for color jpg, 1 for grayscale img
# you need to change it to reflect your dataset
CHANNEL_NUM = 3


# The script assumes that under train_root, there are separate directories for each class
# of training images.
root = "/home/psm2208/data/GTDB/GTDB1/"
start = timeit.default_timer()

def task(path):

    pixel_num = 0  # store all pixel number in the dataset
    channel_sum = np.zeros(CHANNEL_NUM)
    channel_sum_squared = np.zeros(CHANNEL_NUM)

    print('processing image ' + str(path))
    im = cv2.imread(path)  # image in M*N*CHANNEL_NUM shape, channel in BGR order
    im = im / 255.0
    pixel_num += (im.size / CHANNEL_NUM)
    channel_sum += np.sum(im, axis=(0, 1))
    channel_sum_squared += np.sum(np.square(im), axis=(0, 1))

    return (pixel_num, channel_sum, channel_sum_squared)

def cal_dir_stat(root):
    cls_dirs = [d for d in listdir(root) if isdir(join(root, d))]

    paths = []

    for idx, d in enumerate(cls_dirs):
        im_pths = glob(join(root, d, "*.png"))
        paths.extend(im_pths)

    pool = Pool(processes=32)
    ans = pool.map(task, paths)
    pool.close()
    pool.join()

    total_pixel_num = 0
    total_channel_sum = 0
    total_sum_sq = 0

    for pixel_num, channel_sum, channel_sum_squared in ans:
        total_channel_sum += channel_sum
        total_pixel_num += pixel_num
        total_sum_sq += channel_sum_squared

    bgr_mean = total_channel_sum / total_pixel_num
    bgr_std = np.sqrt(total_sum_sq / total_pixel_num - np.square(bgr_mean))

    # change the format from bgr to rgb
    rgb_mean = list(bgr_mean)[::-1]
    rgb_std = list(bgr_std)[::-1]

    return rgb_mean, rgb_std

mean, std = cal_dir_stat(root)
end = timeit.default_timer()
print("elapsed time: {}".format(end - start))
print("mean:{}\nstd:{}".format(mean, std))
