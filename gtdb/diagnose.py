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

# check if two rectangles intersect
def intersects(first, other):
    return not (first[2] < other[0] or
                first[0] > other[2] or
                first[1] > other[3] or
                first[3] < other[1])

def read_data(training_pdf_names, char_dir, gt_math_dir, det_math_dir):

    char_bbs = {}
    args = []
    total_math_char = 0

    for filename in training_pdf_names:

        path = os.path.join(char_dir, filename + ".csv")
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

                if row[6] == 'MATH_SYMBOL':
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

    return training_pdf_names, total_math_char, gt_math_bbs, det_math_bbs, char_bbs


def char_level_eval(training_pdf_names, total_math_char, gt_math_bbs, det_math_bbs, char_bbs):

    args = []

    for key in det_math_bbs:
        for page in det_math_bbs[key]:
            if page not in gt_math_bbs[key]:
                gt_math = []
            else:
                gt_math = gt_math_bbs[key][page]
            args.append([key, det_math_bbs[key][page], char_bbs[key][page], gt_math])

    pool = Pool(processes=16)
    ans = pool.map(character_level_score, args)
    pool.close()
    pool.join()

    detected_math_char = 0
    detected_text_char = 0

    for math, text in ans:
        detected_math_char = detected_math_char + math
        detected_text_char = detected_text_char + text

    print('detected math chars ', detected_math_char)
    print('detected text chars ', detected_text_char)
    print('total math chars ', total_math_char)

    recall = detected_math_char / total_math_char
    precision = detected_math_char / (detected_math_char + detected_text_char)

    fscore =  2 * recall * precision / (recall + precision)

    print('Char Recall\t', recall)
    print('Char Precision\t', precision)
    print('Char F-Score\t', fscore)


def character_level_score(args):

    filename, det_math_bbs, char_bbs, gt_math_bbs = args
    detected_math_char_count = 0
    text_char_count = 0

    for char_info in char_bbs:
        char_bb = [float(char_info[1]), float(char_info[2]), float(char_info[3]), float(char_info[4])]

        for current_math_bb in det_math_bbs:

            math_bb = [float(current_math_bb[1]),float(current_math_bb[2]),
                       float(current_math_bb[3]),float(current_math_bb[4])]

            if box_utils.check_inside(char_bb, math_bb): #TODO

                if char_info[6] == 'MATH_SYMBOL':
                    detected_math_char_count = detected_math_char_count + 1
                    break
                else:
                    text_char_count = text_char_count + 1

    return detected_math_char_count, text_char_count

def box_level_granular_eval(training_pdf_names, total_math_char, gt_math_dir, det_math_dir,
                            gt_math_bbs, det_math_bbs, char_bbs, test_gt_math_dir):

    _, _, detailed_detections = IOUevaluater.IOUeval(test_gt_math_dir, det_math_dir)
    assign_chars_to_math_boxes(gt_math_bbs, char_bbs)
    assign_chars_to_math_boxes(det_math_bbs, char_bbs)

    single_char_det = [0, 0]
    multi_char_det = [0, 0]

    total_single_char_det = 0
    total_multi_char_det = 0

    single_char_gt = 0
    multi_char_gt = 0

    for filename in training_pdf_names:
        current_det = detailed_detections[filename]

        for page in current_det[0]:
            coarse = current_det[0][page]
            fine = current_det[1][page]

            #DET for recall
            for det in coarse:
                if gt_math_bbs[filename][str(int(float(page)))][int(det[3:])-1][5] > 1:
                    multi_char_det[0] = multi_char_det[0] + 1
                else:
                    single_char_det[0] = single_char_det[0] + 1

            for det in fine:
                if gt_math_bbs[filename][str(int(float(page)))][int(det[3:])-1][5] > 1:
                    multi_char_det[1] = multi_char_det[1] + 1
                else:
                    single_char_det[1] = single_char_det[1] + 1

            # DET for precision
            for det in det_math_bbs[filename][str(int(float(page)))]:
                if det[5] > 1:
                    total_multi_char_det = total_multi_char_det + 1
                else:
                    total_single_char_det = total_single_char_det + 1

            #TODO
            # for gt in gt_math_bbs[filename][str(int(float(page)))]:
            #     if gt[5] == 1:
            #         single_char_gt = single_char_gt + 1
            #     else:
            #         multi_char_gt = multi_char_gt + 1

        # GT
        for page in gt_math_bbs[filename]:
           for gt in gt_math_bbs[filename][str(int(float(page)))]:
                if gt[5] > 1:
                    multi_char_gt = multi_char_gt + 1
                else:
                    single_char_gt = single_char_gt + 1

    # single char scores - coarse
    # precision
    print("Number of single character regions correctly detected IOU50, IOU75 ", single_char_det)
    print("Total number of single character regions detected ", total_single_char_det)
    print("Total number of single character regions GT ", single_char_gt)

    print("Number of multi character regions correctly detected IOU50, IOU75 ", multi_char_det)
    print("Total number of multi character regions detected ", total_multi_char_det)
    print("Total number of multi character regions GT ", multi_char_gt)

    # Single character regions

    print("***** Results : Single Character Regions ***** ")
    prec_50 = single_char_det[0]/total_single_char_det
    rec_50 = single_char_det[0] / single_char_gt
    fscore_50 = 2*prec_50*rec_50/(prec_50 + rec_50)

    print("Precision IOU50 ", prec_50)
    print("Recall IOU50 ", rec_50)
    print("F-score IOU50 ", fscore_50)

    prec_75 = single_char_det[1] / total_single_char_det
    rec_75 = single_char_det[1] / single_char_gt
    fscore_75 = 2 * prec_75 * rec_75 / (prec_75 + rec_75)

    print("Precision IOU75 ", prec_75)
    print("Recall IOU75 ", rec_75)
    print("F-score IOU75 ", fscore_75)

    print("***** Results : Multi Character Regions ***** ")
    prec_50 = multi_char_det[0] / total_multi_char_det
    rec_50 = multi_char_det[0] / multi_char_gt
    fscore_50 = 2 * prec_50 * rec_50 / (prec_50 + rec_50)

    print("Precision IOU50 ", prec_50)
    print("Recall IOU50 ", rec_50)
    print("F-score IOU50 ", fscore_50)

    prec_75 = multi_char_det[1] / total_multi_char_det
    rec_75 = multi_char_det[1] / multi_char_gt
    fscore_75 = 2 * prec_75 * rec_75 / (prec_75 + rec_75)

    print("Precision IOU75 ", prec_75)
    print("Recall IOU75 ", rec_75)
    print("F-score IOU75 ", fscore_75)

def find_merged_regions(training_pdf_names, gt_math_boxes, det_math_boxes):

    det_regions_with_multi_math = 0

    for pdf_name in training_pdf_names:
        for page in det_math_boxes[pdf_name]:

            for det in det_math_boxes[pdf_name][page]:

                count = 0

                det_bb = [float(det[1]), float(det[2]),
                          float(det[3]), float(det[4])]

                if page not in gt_math_boxes[pdf_name]:
                    continue

                for gt in gt_math_boxes[pdf_name][page]:

                    gt_bb = [float(gt[1]),float(gt[2]),
                             float(gt[3]),float(gt[4])]

                    if box_utils.check_inside(gt_bb, det_bb):
                        count = count + 1

                    if count > 1:
                        det_regions_with_multi_math = \
                            det_regions_with_multi_math + count
                        break

    print("Merged boxes ", det_regions_with_multi_math)


def assign_chars_to_math_boxes(all_math_boxes, all_char_bbs):

    for pdf_name in all_math_boxes:
        for page in all_math_boxes[pdf_name]:

            #print('Assigning ', pdf_name, page)
            math_boxes = all_math_boxes[pdf_name][page]
            char_bbs = all_char_bbs[pdf_name][str(int(float(page)))]

            for math_box in math_boxes:
                math_box.append(0)

            for char_info in char_bbs:
                for math_bb in math_boxes:

                    current_char_bb = [float(char_info[1]), float(char_info[2]), #TODO index from 1
                                       float(char_info[3]), float(char_info[4])]

                    current_math_bb = [float(math_bb[1]),float(math_bb[2]),
                                       float(math_bb[3]),float(math_bb[4])]

                    if box_utils.check_inside(current_char_bb, current_math_bb):
                        math_bb[-1] = math_bb[-1] + 1


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
    test_gt_math_dir = sys.argv[5] #/home/psm2208/Workspace/Task3_Detection/Test/test_math/

    image_dir = '/home/psm2208/data/GTDB/images/'

    training_pdf_names, total_math_char, gt_math_bbs, det_math_bbs, char_bbs = \
        read_data(training_pdf_names_list, gt_char_dir, gt_math_dir, detected_math_dir)

    char_level_eval(training_pdf_names, total_math_char, copy.deepcopy(gt_math_bbs),
                    copy.deepcopy(det_math_bbs), copy.deepcopy(char_bbs))

    box_level_granular_eval(training_pdf_names, total_math_char, gt_math_dir,
                            detected_math_dir, gt_math_bbs,
                            det_math_bbs, char_bbs,test_gt_math_dir)

    find_merged_regions(training_pdf_names, copy.deepcopy(gt_math_bbs), copy.deepcopy(det_math_bbs))
