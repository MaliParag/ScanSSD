# Author: Parag Mali
# This file contains functions that calculate character level detection results

import os
import sys
sys.path.extend(['/home/psm2208/code', '/home/psm2208/code'])

import csv
from multiprocessing import Pool
from IOU_lib import IOUevaluater
import copy
from gtdb import box_utils
import numpy as np

def read_data(training_pdf_names, char_dir, gt_math_dir):

    char_bbs = {}

    for filename in training_pdf_names:

        path = os.path.join(char_dir, filename + ".char")
        print('Processing ' + path)
        count = 0

       #data = adjust_data[filename]

        map = {}
        with open(path, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                #print('row is ' + str(row[1]))
                # if entry is not in map
                if str(int(float(row[0]))) not in map:
                    map[str(int(float(row[0])))] = []

                map[str(int(float(row[0])))].append(row)
                count = count + 1

        char_bbs[filename] = map

    gt_math_bbs = {}

    for filename in training_pdf_names:

        path = os.path.join(gt_math_dir, filename + ".csv")

        map = {}
        with open(path, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                # if entry is not in map
                if str(int(float(row[0]))) not in map:
                    map[str(int(float(row[0])))] = []

                map[str(int(float(row[0])))].append(row)

        gt_math_bbs[filename] = map

    return training_pdf_names, gt_math_bbs, char_bbs


def char_level_eval(training_pdf_names, gt_math_bbs, char_bbs, output_dir):

    args = []

    for key in char_bbs:
        for page in char_bbs[key]:
            if page not in gt_math_bbs[key]:
                gt_math = []
            else:
                gt_math = gt_math_bbs[key][page]
            args.append([key, gt_math, char_bbs[key][page], page, output_dir])

    pool = Pool(processes=4)
    ans = pool.map(find_word_bbs, args)
    pool.close()
    pool.join()

def assign_type_char(char_bbs, gt_math_bbs):

    updated_char_bbs = []
    for char_info in char_bbs:
        char_bb = [float(char_info[1]), float(char_info[2]), float(char_info[3]), float(char_info[4])]

        for current_math_bb in gt_math_bbs:

            math_bb = [float(current_math_bb[1]), float(current_math_bb[2]),
                       float(current_math_bb[3]), float(current_math_bb[4])]

            if box_utils.check_inside(char_bb, math_bb) or \
               char_info[5] == 'MATH_SYMBOL':
                char_info[5] = 'MATH_SYMBOL'

        updated_char_bbs.append(char_info)

    return updated_char_bbs

def find_word_bbs(args):

    filename, gt_math_bbs, char_bbs, page, output_dir = args
    char_bbs = assign_type_char(char_bbs, gt_math_bbs)

    word_bbs = []

    word_bb = []
    for char_info in char_bbs:
        current_char_bb = [float(char_info[1]), float(char_info[2]),
                           float(char_info[3]), float(char_info[4])]

        if char_info[5] == 'MATH_SYMBOL' or char_info[-1] in ['0E81']:#, '142E']:
            if len(word_bb) == 4:
                word_bbs.append(word_bb)
            word_bb = []
        elif len(word_bb) == 4 and (current_char_bb[0] <= word_bb[2] or current_char_bb[1] > word_bb[3]):
             word_bbs.append(word_bb)
             word_bb = current_char_bb
        else:
            if len(word_bb) != 4:
                word_bb = current_char_bb
            else:
                word_bb = box_utils.merge(word_bb, current_char_bb)

    # add last word
    if len(word_bb) == 4:
        word_bbs.append(word_bb)

    # merge bbs
    final_word_bbs = []
    word_bb = []

    try:
        # merge intersection word bbs
        for i in range(0, len(word_bbs)):
            if len(word_bb) != 4:
                word_bb = word_bbs[i]
            else:
                if box_utils.intersects(word_bb, word_bbs[i]):
                    word_bb = box_utils.merge(word_bb, word_bbs[i])
                else:
                    final_word_bbs.append(word_bb)
                    word_bb = word_bbs[i]

        final_word_bbs.append(word_bb)
    except Exception as e:
        print("Exception while processing ", filename, " ", page, " ", sys.exc_info(), e)
        print('word ', len(word_bb))
        print(len(word_bbs[i]))

    if not os.path.exists(os.path.dirname(output_dir)):
        os.makedirs(os.path.dirname(output_dir))

    word_file_path = os.path.join(output_dir, filename + ".csv")
    word_file = open(word_file_path, 'a')

    word_bbs = np.array(final_word_bbs)
    page_nos = [[int(page)]] * len(word_bbs)
    word_bbs = np.append(page_nos, word_bbs, axis=1)
    np.savetxt(word_file, word_bbs, fmt='%.2f', delimiter=',')
    word_file.close()


if __name__ == '__main__':

    training_pdf_names = open(sys.argv[1], 'r')

    training_pdf_names_list = []

    # for each training image pdf file
    for pdf_name in training_pdf_names:
        pdf_name = pdf_name.strip()
        if pdf_name != '':
            training_pdf_names_list.append(pdf_name)
    training_pdf_names.close()

    gt_math_dir = sys.argv[2]
    gt_char_dir = sys.argv[3]
    output_dir = sys.argv[4]

    training_pdf_names, gt_math_bbs, char_bbs = \
        read_data(training_pdf_names_list, gt_char_dir, gt_math_dir)

    char_level_eval(training_pdf_names, copy.deepcopy(gt_math_bbs), copy.deepcopy(char_bbs), output_dir)
