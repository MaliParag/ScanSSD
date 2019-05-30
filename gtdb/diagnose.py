# Author: Parag Mali
# This script divides a big image into smaller images using a sliding window.
# It also divides corresponding bounding box annotations.

# read the image
import os
import sys
import csv
from multiprocessing import Pool
from IOU_lib import IOUevaluater

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

        path = os.path.join(char_dir, filename + ".char")

        map = {}
        with open(path, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                # if entry is not in map
                if row[0] not in map:
                    map[row[0]] = []

                if row[6] == 'MATH_SYMBOL':
                    total_math_char = total_math_char + 1

                map[row[0]].append(row)

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
                if row[0] not in map:
                    map[row[0]] = []


                map[row[0]].append(row)

        det_math_bbs[filename] = map

    gt_math_bbs = {}

    for filename in training_pdf_names:

        path = os.path.join(gt_math_dir, filename + ".math")

        map = {}
        with open(path, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                # if entry is not in map
                if row[0] not in map:
                    map[row[0]] = []

                map[row[0]].append(row)

        gt_math_bbs[filename] = map

    return training_pdf_names, total_math_char, gt_math_bbs, det_math_bbs, char_bbs


def char_level_eval(training_pdf_names, total_math_char, gt_math_bbs, det_math_bbs, char_bbs):

    args = []

    for key in det_math_bbs:
        for page in det_math_bbs[key]:
            args.append([key, det_math_bbs[key][page], char_bbs[key][page], gt_math_bbs[key][page]])

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

    print('Recall\t', recall)
    print('Precision\t', precision)
    print('F-Score\t', fscore)


def character_level_score(args):

    filename, det_math_bbs, char_bbs, gt_math_bbs = args
    detected_char_count = 0
    text_char_count = 0

    for char_info in char_bbs:
        char_bb = [float(char_info[2]), float(char_info[3]), float(char_info[4]), float(char_info[5])]

        for current_math_bb in det_math_bbs:

            math_bb = [float(current_math_bb[1]),float(current_math_bb[2]),
                       float(current_math_bb[3]),float(current_math_bb[4])]

            if intersects(char_bb, math_bb):

                if char_info[6] == 'MATH_SYMBOL':
                    detected_char_count = detected_char_count + 1
                    break
                else:
                    text_char_count = text_char_count + 1

    return detected_char_count, text_char_count

def box_level_granular_eval(training_pdf_names, total_math_char, gt_math_dir, det_math_dir,
                            gt_math_bbs, det_math_bbs, char_bbs):

    _, _, detailed_detections = IOUevaluater.IOUeval(gt_math_dir, det_math_dir)
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
                if gt_math_bbs[filename][str(page)][int(det[3:])-1][5] == 1:
                    single_char_det[0] = single_char_det[0] + 1
                else:
                    multi_char_det[0] = multi_char_det[0] + 1

            for det in fine:
                if gt_math_bbs[filename][str(page)][int(det[3:])-1][5] == 1:
                    single_char_det[1] = single_char_det[1] + 1
                else:
                    multi_char_det[1] = multi_char_det[1] + 1

            # DET for precision
            for det in det_math_bbs[filename][str(page)]:
                if det[5] == 1:
                    total_single_char_det = total_single_char_det + 1
                else:
                    total_multi_char_det = total_multi_char_det + 1

            # GT
            for gt in gt_math_bbs[filename][str(page)]:
                if gt[5] == 1:
                    single_char_gt = single_char_gt + 1
                else:
                    multi_char_gt = multi_char_gt + 1


    # single char scores - coarse
    # precision
    print("Number of single character regions correctly detected @50,@75", single_char_det)
    print("Total number of single character regions detected", total_single_char_det)
    print("Total number of single character regions GT ", single_char_gt)

    print("Number of multi character regions correctly detected @50,@75", multi_char_det)
    print("Total number of multi character regions detected", total_multi_char_det)
    print("Total number of multi character regions GT ", multi_char_gt)

    # Single character regions

    print("***** Results : Single Character Regions ***** ")
    prec_50 = single_char_det[0]/total_single_char_det
    rec_50 = single_char_det[0] / single_char_gt
    fscore_50 = 2*prec_50*rec_50/(prec_50 + rec_50)

    print("Precision@50", prec_50)
    print("Recall@50", rec_50)
    print("F-score@50", fscore_50)

    prec_75 = single_char_det[1] / total_single_char_det
    rec_75 = single_char_det[1] / single_char_gt
    fscore_75 = 2 * prec_75 * rec_75 / (prec_75 + rec_75)

    print("Precision@75", prec_75)
    print("Recall@75", rec_75)
    print("F-score@75", fscore_75)

    print("***** Results : Multi Character Regions ***** ")
    prec_50 = multi_char_det[0] / total_multi_char_det
    rec_50 = multi_char_det[0] / multi_char_gt
    fscore_50 = 2 * prec_50 * rec_50 / (prec_50 + rec_50)

    print("Precision@50", prec_50)
    print("Recall@50", rec_50)
    print("F-score@50", fscore_50)

    prec_75 = multi_char_det[1] / total_multi_char_det
    rec_75 = multi_char_det[1] / multi_char_gt
    fscore_75 = 2 * prec_75 * rec_75 / (prec_75 + rec_75)

    print("Precision@75", prec_75)
    print("Recall@75", rec_75)
    print("F-score@75", fscore_75)

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

                    if intersects(det_bb, gt_bb):
                        count = count + 1

                if count > 1:
                    det_regions_with_multi_math = \
                        det_regions_with_multi_math + count

    print("Merged boxes ", det_regions_with_multi_math)


def assign_chars_to_math_boxes(all_math_boxes, all_char_bbs):

    for pdf_name in all_math_boxes:
        for page in all_math_boxes[pdf_name]:

            math_boxes = all_math_boxes[pdf_name][page]
            char_bbs = all_char_bbs[pdf_name][page]

            for math_box in math_boxes:
                math_box.append(0)

            for char_info in char_bbs:
                for math_bb in math_boxes:

                    current_char_bb = [float(char_info[2]), float(char_info[3]),
                                       float(char_info[4]), float(char_info[5])]

                    current_math_bb = [float(math_bb[1]),float(math_bb[2]),
                                       float(math_bb[3]),float(math_bb[4])]

                    if intersects(current_char_bb, current_math_bb):
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

    gt_char_dir = '/home/psm2208/data/GTDB/char_annotations/'
    gt_math_dir = '/home/psm2208/data/GTDB/annotations/'
    detected_math_dir = '/home/psm2208/code/eval/final_submission/Test/'
    image_dir = '/home/psm2208/data/GTDB/images/'

    training_pdf_names, total_math_char, gt_math_bbs, det_math_bbs, char_bbs = \
        read_data(training_pdf_names_list, gt_char_dir, gt_math_dir, detected_math_dir)

    #char_level_eval(training_pdf_names, total_math_char, gt_math_bbs, det_math_bbs, char_bbs)
    #box_level_granular_eval(training_pdf_names, total_math_char, gt_math_dir,
    #                        detected_math_dir, gt_math_bbs, det_math_bbs, char_bbs)

    find_merged_regions(training_pdf_names, gt_math_bbs, det_math_bbs)