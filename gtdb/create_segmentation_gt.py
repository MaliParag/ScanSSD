# Rectangles after projection

import sys
sys.path.extend(['/home/psm2208/code', '/home/psm2208/code'])
import os
import csv
import numpy as np
from multiprocessing import Pool
import shutil
from gtdb import feature_extractor

def intersects(first, other):
    return not (first[2] < other[0] or
                first[0] > other[2] or
                first[1] > other[3] or
                first[3] < other[1])


def create_gt(args):
    count = 0

    try:
        output_dir, pdf_name, page_num, gt_page_math, det_page_math = args


        inside_gt_dict = {}

        # for each det i, find the gt with which it intersects
        for i, det in enumerate(det_page_math):

            inside_gt_dict[i] = set()

            for j, gt in enumerate(gt_page_math):
                if intersects(det, gt):
                    inside_gt_dict[i].add(j)

        check_dict = {}
        for i, gt in enumerate(gt_page_math):

            check_dict[i] = set()

            for j, det in enumerate(det_page_math):
                if check_inside(det, gt):
                    check_dict[i].add(j)

        for key in check_dict:
            if len(check_dict[key]) > 1:
                count = count + 1

        segmentation_gt = []

        for i, det_math1 in enumerate(det_page_math):

            min = float('inf')
            min_idx = -1

            x1 = det_math1[0] + ((det_math1[2] - det_math1[0]) / 2)
            y1 = det_math1[1] + ((det_math1[3] - det_math1[1]) / 2)

            for j, det_math in enumerate(det_page_math):
                if i != j:

                    x2 = det_math[0] + ((det_math[2] - det_math[0]) / 2)
                    y2 = det_math[1] + ((det_math[3] - det_math[1]) / 2)

                    c_dist = np.sqrt((y2 - y1) * (y2 - y1) + (x2 - x1) * (x2 - x1))#feature_extractor.vertical_dist_bb(det_page_math[i], det_page_math[j])

                    if c_dist < min:
                        min = c_dist
                        min_idx = j

            if len(inside_gt_dict[i].intersection(inside_gt_dict[min_idx])) > 0:
                # positive example
                segmentation_gt.append(
                    feature_extractor.extract_features(det_page_math[i], det_page_math[min_idx], 1))
            else:
                #negative example
                segmentation_gt.append(
                    feature_extractor.extract_features(det_page_math[i], det_page_math[min_idx], 0))

        output_file = os.path.join(output_dir, "gt.csv")
        writer = csv.writer(open(output_file,"a"), delimiter=",")

        for gt_row in segmentation_gt:
            writer.writerow(gt_row)

        print('Processed ', pdf_name, ' ', page_num)

    except:
        print("Exception while processing ", pdf_name, " ", page_num, " ", sys.exc_info())

    return count

def create_gt_segmentation(filename, gt_math_dir, det_math_dir, output_dir):

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    pages_list = []
    pdf_names = open(filename, 'r')

    for pdf_name in pdf_names:
        print('Processing-1', pdf_name)
        pdf_name = pdf_name.strip()

        if pdf_name != '':
            gt_math_file = os.path.join(gt_math_dir, pdf_name + ".csv")
            gt_math_regions = np.genfromtxt(gt_math_file, delimiter=',', dtype=int)

            det_math_file = os.path.join(det_math_dir, pdf_name + ".csv")
            det_math_regions = np.genfromtxt(det_math_file, delimiter=',', dtype=int)

            pages = np.unique(gt_math_regions[:, 0])

            for page_num in pages:

                gt_page_math = gt_math_regions[np.where(gt_math_regions[:,0]==page_num)]
                gt_page_math = gt_page_math[:,1:]

                det_page_math = det_math_regions[np.where(det_math_regions[:, 0] == page_num)]
                det_page_math = det_page_math[:, 1:]

                pages_list.append([output_dir, pdf_name, page_num, gt_page_math, det_page_math])

    pdf_names.close()

    pool = Pool(processes=1)
    result = pool.map(create_gt, pages_list)
    pool.close()
    pool.join()
    print('Merged regions', np.sum(result))


def check_inside(rectA, rectB):

    # returns True if A is inside B
    #left, top, right, bottom
    #If any of the sides from A are outside of B
    if rectA[3] < rectB[1]: # if bottom of rectA is less than top of rectB
        return False
    if rectA[1] > rectB[3]: # if top of rectA is greater than bottom of rectB
        return False
    if rectA[2] < rectB[0]: # if right of rectA is less than left of rectB
        return False
    if rectA[0] > rectB[2]: # if left of rectangleA is greater than right of rectB
        return False

    #If none of the sides from A are outside B
    return True

if __name__ == "__main__":
    home_data = "/home/psm2208/data/GTDB/"
    home_eval = "/home/psm2208/code/eval/"
    home_images = "/home/psm2208/data/GTDB/images/"
    home_anno = "/home/psm2208/data/GTDB/annotations/"
    home_char = "/home/psm2208/data/GTDB/char_annotations/"

    output_dir = "/home/psm2208/code/eval/segmentation_gt/"
    gt_math = "/home/psm2208/Workspace/Task3_Detection/Train/GT_math_csv/"

    det_math = "/home/psm2208/code/eval/Train3_Focal_10_25/equal_30.0"

    type = sys.argv[1]


    #filename, gt_math_dir, det_math_dir, output_dir
    create_gt_segmentation(home_data + type, gt_math, det_math, output_dir)

