# Author: Parag Mali
# This script reads ground truths and normalizes them using image size

# read the image
import sys
sys.path.extend(['/home/psm2208/code', '/home/psm2208/code'])
import cv2
import os
import numpy as np
from multiprocessing import Pool
from gtdb import fit_box
from gtdb import feature_extractor
import argparse

# Default parameters for thr GTDB dataset
def parse_args():

    parser = argparse.ArgumentParser(
        description='Stitching method')

    parser.add_argument('--data_file', default='test',
                        type=str, help='choose one')
    parser.add_argument('--output_dir', default='.',
                        help='Output directory path')
    parser.add_argument('--math_dir', required=True,
                        type=str, help='detections dir')
    parser.add_argument('--math_ext', default='.csv',
                        help='Extention of detection files')
    parser.add_argument('--home_data', default='/home/psm2208/data/GTDB/', type = str,
                        help='database dir')
    parser.add_argument('--home_eval', default='/home/psm2208/code/eval/', type = str,
                        help='Eval dir')
    parser.add_argument('--home_images', default='/home/psm2208/data/GTDB/images/', type = str,
                        help='Images dir')
    parser.add_argument('--home_anno', default='/home/psm2208/data/GTDB/annotations/', type = str,
                        help='Annotations dir')
    parser.add_argument('--home_char', default='/home/psm2208/data/GTDB/char_annotations/', type = str,
                        help='Char anno dir')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of workers')

    return parser.parse_args()

def read_math(args, pdf_name):

    math_file = os.path.join(args.math_dir, pdf_name + args.math_ext)
    data = np.array([])

    if os.path.exists(math_file):
        data = np.genfromtxt(math_file, delimiter=',')

        # if there is only one entry convert it to correct form required
        if len(data.shape) == 1:
            data = data.reshape(1, -1)

    if args.math_ext == '.char':
        data = np.delete(data,1,1)
        data = data[:,:5]

    return data.astype(int)

def normalize(params):

    args, math_regions, pdf_name, page_num = params
    print('Processing ', pdf_name, ' > ', page_num)

    image = cv2.imread(os.path.join(args.home_images, pdf_name, str(int(page_num + 1)) + ".png"))
    im_bw = fit_box.convert_to_binary(image)

    new_math = []
    for math in math_regions:
        box = [math[0]/im_bw.shape[1], math[1]/im_bw.shape[0],
               math[2]/im_bw.shape[1], math[3]/im_bw.shape[0]]#fit_box.adjust_box(im_bw, math)

        if feature_extractor.width(box) > 0 and feature_extractor.height(box) > 0:
            new_math.append(box)

    return new_math

def normalize_boxes(args):
    pdf_list = []
    pdf_names_file = open(args.data_file, 'r')

    for pdf_name in pdf_names_file:
        pdf_name = pdf_name.strip()

        if pdf_name != '':
            pdf_list.append(pdf_name)

    math_regions = {}

    for pdf_name in pdf_list:
        math_regions[pdf_name] = read_math(args, pdf_name)

    voting_ip_list = []
    for pdf_name in pdf_list:

        pages = np.unique(math_regions[pdf_name][:, 0])

        #args, math_regions, pdf_name, page_num
        for page_num in pages:
            current_math = math_regions[pdf_name][math_regions[pdf_name][:,0] == page_num]
            voting_ip_list.append([args, np.delete(current_math, 0, 1), pdf_name, page_num])

    pool = Pool(processes=args.num_workers)
    out = pool.map(normalize, voting_ip_list)

    for ip, final_math in zip(voting_ip_list, out):
        pdf_name = ip[2]
        page_num = ip[3]

        col = np.array([int(page_num)] * len(final_math))
        final_math = np.concatenate((col[:, np.newaxis], final_math), axis=1)

        math_file_path = os.path.join(args.output_dir, pdf_name + '.csv')

        if not os.path.exists(os.path.dirname(math_file_path)):
            os.makedirs(os.path.dirname(math_file_path))

        math_file = open(math_file_path, 'a')

        np.savetxt(math_file, final_math, fmt='%.2f', delimiter=',')
        math_file.close()

if __name__ == '__main__':

    args = parse_args()
    normalize_boxes(args)
