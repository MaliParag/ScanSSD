# Author: Parag Mali
# This script stitches back the output generated on the image patches (sub-images)

# read the image
import sys
sys.path.extend(['/home/psm2208/code', '/home/psm2208/code'])
import cv2
import os
import csv
import numpy as np
import utils.visualize as visualize
from multiprocessing import Pool
from cv2.dnn import NMSBoxes
from scipy.ndimage.measurements import label
import scipy.ndimage as ndimage
import copy
from gtdb import fit_box
from gtdb import box_utils
from gtdb import feature_extractor
import shutil
import time
from collections import OrderedDict
from collections import deque
import argparse

def parse_args():
    '''
    Parameters
    '''
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
    parser.add_argument('--type', default='math', type=str, help='Math or text')

    return parser.parse_args()

def read_math(args, pdf_name):
    '''
    Read math bounding boxes for given PDF
    '''
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

def read_char(args, pdf_name):
    '''
    Read char bounding boxes for given PDF
    '''
    data = []

    path = os.path.join(args.home_char, pdf_name + ".char")

    with open(path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            #print('row is ' + str(row[1]))
            # if entry is not in map
            data.append(row)

    return np.array(data)


def adjust(params):
    '''
    Fit the bounding boxes to the characters
    '''
    args, math_regions, pdf_name, page_num = params
    print('Processing ', pdf_name, ' > ', page_num)

    image = cv2.imread(os.path.join(args.home_images, pdf_name, str(int(page_num + 1)) + ".png"))
    im_bw = fit_box.convert_to_binary(image)

    new_math = []
    for math in math_regions:
        box = fit_box.adjust_box(im_bw, math)

        if feature_extractor.width(box) > 0 and feature_extractor.height(box) > 0:
            new_math.append(box)

    return new_math

def adjust_char(params):
    '''
    Adjust the character bounding boxes
    '''
    try:
        args, char_regions, pdf_name, page_num = params
        print('Char processing ', pdf_name, ' > ', page_num)

        image = cv2.imread(os.path.join(args.home_images, pdf_name, str(int(page_num) + 1) + ".png"))
        im_bw = fit_box.convert_to_binary(image)

        new_chars = []

        for char in char_regions:

            bb_char = [char[2],char[3],char[4],char[5]]
            bb_char = [int(float(k)) for k in bb_char]
            box = fit_box.adjust_box(im_bw, bb_char)
            if feature_extractor.width(box) > 0 and feature_extractor.height(box) > 0:
                char[1] = box[0]
                char[2] = box[1]
                char[3] = box[2]
                char[4] = box[3]
                new_chars.append(char)

        return new_chars
    except Exception as e:
        print('Error while processing ', pdf_name, ' > ', page_num, sys.exc_info())
        return []

def adjust_boxes(args):
    '''
    Driving function for adjusting the boxes
    '''

    pdf_list = []
    pdf_names_file = open(args.data_file, 'r')

    for pdf_name in pdf_names_file:
        pdf_name = pdf_name.strip()

        if pdf_name != '':
            pdf_list.append(pdf_name)

    regions = {}

    for pdf_name in pdf_list:
        if args.type == 'char':
            regions[pdf_name] = read_char(args, pdf_name)
        else:
            regions[pdf_name] = read_math(args, pdf_name)

    voting_ip_list = []

    for pdf_name in pdf_list:

        pages = np.unique(regions[pdf_name][:, 0])

        #args, math_regions, pdf_name, page_num
        for page_num in pages:

            current_math = regions[pdf_name][regions[pdf_name][:,0] == page_num]

            if args.type == 'math':
                current_math = np.delete(current_math, 0, 1)

            voting_ip_list.append([args, current_math, pdf_name, page_num])

    pool = Pool(processes=args.num_workers)

    if args.type == 'math':
        out = pool.map(adjust, voting_ip_list)
    else:
        out = pool.map(adjust_char, voting_ip_list)

    for ip, final_math in zip(voting_ip_list, out):
        pdf_name = ip[2]
        page_num = ip[3]

        if args.type == 'math':
            col = np.array([int(page_num)] * len(final_math))
            final_math = np.concatenate((col[:, np.newaxis], final_math), axis=1)

        math_file_path = os.path.join(args.output_dir, pdf_name + '.csv')

        if not os.path.exists(os.path.dirname(math_file_path)):
            os.makedirs(os.path.dirname(math_file_path))


        if args.type == 'math':
            math_file = open(math_file_path, 'a')

            np.savetxt(math_file, final_math, fmt='%.2f', delimiter=',')
            math_file.close()

        else:

            with open(math_file_path, 'a') as csvfile:
                writer = csv.writer(csvfile, delimiter=",")

                for math_region in final_math:
                    writer.writerow(math_region)


if __name__ == '__main__':

    args = parse_args()
    adjust_boxes(args)
