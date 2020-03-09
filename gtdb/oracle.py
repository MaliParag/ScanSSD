# Author: Parag Mali
# This file contains functions that calculate character level detection results

import os
import sys
import cv2
sys.path.extend(['/home/psm2208/code', '/home/psm2208/code'])

import csv
from multiprocessing import Pool
from IOU_lib import IOUevaluater
import copy
from gtdb import box_utils

# check if two rectangles intersect
def intersects(first, other):
    return not (first[2] < other[0] or
                first[0] > other[2] or
                first[1] > other[3] or
                first[3] < other[1])

def read_data(training_pdf_names, char_dir, gt_math_dir, det_math_dir, word_bb_dir):

    char_bbs = {}
    args = []
    total_math_char = 0
    word_bbs = {}

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

                if row[5] == 'MATH_SYMBOL':
                    total_math_char = total_math_char + 1

                # row[2] = data[count][1]
                # row[3] = data[count][2]
                # row[4] = data[count][3]
                # row[5] = data[count][4]

                map[str(int(float(row[0])))].append(row)
                count = count + 1

        char_bbs[filename] = map

    det_math_bbs = {}

    for filename in training_pdf_names:

        #path = os.path.join(math_dir, filename + ".math")
        path = os.path.join(det_math_dir, filename + ".csv")

        map = {}
        with open(path, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                # if entry is not in map
                if str(int(float(row[0]))) not in map:
                    map[str(int(float(row[0])))] = []

                map[str(int(float(row[0])))].append(row)

        det_math_bbs[filename] = map

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

    for filename in training_pdf_names:

        path = os.path.join(word_bb_dir, filename + ".csv")

        map = {}
        with open(path, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                # if entry is not in map
                if str(int(float(row[0]))) not in map:
                    map[str(int(float(row[0])))] = []

                map[str(int(float(row[0])))].append(row)

        word_bbs[filename] = map

    return training_pdf_names, total_math_char, gt_math_bbs, det_math_bbs, char_bbs, word_bbs

def char_level_eval(image_dir, training_pdf_names, total_math_char, gt_math_bbs,
                    det_math_bbs, char_bbs, word_bbs, op_image_dir):

    args = []

    for key in det_math_bbs:
        for page in det_math_bbs[key]:
            if page not in gt_math_bbs[key]:
                gt_math = []
            else:
                gt_math = gt_math_bbs[key][page]
            args.append([key, det_math_bbs[key][page], char_bbs[key][page], gt_math, word_bbs[key][page],
                         page, image_dir, op_image_dir])

    pool = Pool(processes=16)
    ans = pool.map(character_level_score, args)
    pool.close()
    pool.join()

    total_detected = 0
    total_undetected = 0
    total_incorrect = 0

    for vals in ans:
        total_detected = total_detected + vals[0]
        total_incorrect = total_incorrect + vals[1]

    print('Total detected ', total_detected)
    # print('Total undetected', total_undetected)
    print('Total incorrect', total_incorrect)


def character_level_score(args):

    filename, det_math_bbs, char_bbs, gt_math_bbs, word_bbs, page_num, image_dir, op_image_dir = args

    page_num = str(int(page_num) + 1)
    image_file = os.path.join(image_dir, filename, page_num + ".png")
    image = cv2.imread(image_file)

    op_image_file = os.path.join(op_image_dir, filename, page_num + ".png")

    if not os.path.exists(os.path.dirname(op_image_file)):
        os.makedirs(os.path.dirname(op_image_file))

    is_det_char_bbs = []
    detected_math_exp_count = 0
    incorrectly_det_math_exp_count = 0
    current_total = 0

    chars_found = 0

    # find which characters are detected by our method
    for char_bb in char_bbs:
        found = False
        current_char_bb = [float(char_bb[1]), float(char_bb[2]), float(char_bb[3]), float(char_bb[4])]
        for det_math_bb in det_math_bbs:
            current_det = [float(det_math_bb[1]), float(det_math_bb[2]), float(det_math_bb[3]), float(det_math_bb[4])]
            if box_utils.check_inside(current_char_bb, current_det):
                found = True
                chars_found = chars_found + 1
                break
        is_det_char_bbs.append(found)

        if (char_bb[5] != 'MATH_SYMBOL' and found):
            cv2.rectangle(image, (int(current_char_bb[0]), int(current_char_bb[1])),
                                 (int(current_char_bb[2]), int(current_char_bb[3])), (0, 255, 0), 3)


    # find the oracle metric mentioned in Unet paper by Uchida
    for gt_math_bb in gt_math_bbs:
        current_total = 0
        current_det = 0
        current_gt = [float(gt_math_bb[1]), float(gt_math_bb[2]), float(gt_math_bb[3]), float(gt_math_bb[4])]
        for idx, char_bb in enumerate(char_bbs):
            current_char_bb = [float(char_bb[1]), float(char_bb[2]), float(char_bb[3]), float(char_bb[4])]
            if box_utils.check_inside(current_char_bb, current_gt):
                current_total = current_total + 1
                if is_det_char_bbs[idx]:
                    current_det = current_det + 1

        if current_det > (current_total/2):
            detected_math_exp_count = detected_math_exp_count + 1

        #cv2.rectangle(image, (int(current_gt[0]), int(current_gt[1])),
        #              (int(current_gt[2]), int(current_gt[3])), (0, 0, 255), 3)

    #cv2.imwrite(op_image_file, image)
    #undetected_math_exp_count = current_total - detected_math_exp_count

    # count incorrectly_det_math_exp_count
    for word_bb in word_bbs:
        current_total = 0
        current_det = 0
        current_word_bb = [float(word_bb[1]), float(word_bb[2]), float(word_bb[3]), float(word_bb[4])]
        for idx, char_bb in enumerate(char_bbs):
            current_char_bb = [float(char_bb[1]), float(char_bb[2]), float(char_bb[3]), float(char_bb[4])]
            if box_utils.check_inside(current_char_bb, current_word_bb):
                current_total = current_total + 1
                if is_det_char_bbs[idx]:
                    current_det = current_det + 1

        if current_det > (current_total/2):
            incorrectly_det_math_exp_count = incorrectly_det_math_exp_count + 1

        #cv2.rectangle(image, (int(current_gt[0]), int(current_gt[1])),
        #              (int(current_gt[2]), int(current_gt[3])), (0, 0, 255), 3)

    return detected_math_exp_count, incorrectly_det_math_exp_count


if __name__ == '__main__':

    training_pdf_names = open(sys.argv[1], 'r')

    training_pdf_names_list = []

    # for each training image pdf file
    for pdf_name in training_pdf_names:
        pdf_name = pdf_name.strip()
        if pdf_name != '':
            training_pdf_names_list.append(pdf_name)
    training_pdf_names.close()

    detected_math_dir = sys.argv[2] #'/home/psm2208/code/eval/final_submission/Test/'
    gt_math_dir = sys.argv[3]  # '/home/psm2208/data/GTDB/annotations/'
    gt_char_dir = sys.argv[4]#'/home/psm2208/data/GTDB/char_annotations/'
    image_dir = sys.argv[5] #'/home/psm2208/data/GTDB/images/'
    op_image_dir = sys.argv[6] # op image

    word_bb_dir = sys.argv[7] # word bb dir

    training_pdf_names, total_math_char, gt_math_bbs, det_math_bbs, char_bbs, word_bbs = \
        read_data(training_pdf_names_list, gt_char_dir, gt_math_dir, detected_math_dir, word_bb_dir)

    char_level_eval(image_dir, training_pdf_names, total_math_char, copy.deepcopy(gt_math_bbs),
                    copy.deepcopy(det_math_bbs), copy.deepcopy(char_bbs), copy.deepcopy(word_bbs),
                    op_image_dir)
